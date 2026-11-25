#!/usr/bin/env python3
"""main.py -- a fruit-fly connectome plays Breath of the Wild through a Windows named-pipe bridge.

    game  --telemetry-->  \\\\.\\pipe\\botw_fly_bridge  -->  botw_encoder  -->  FlyBrain  -->  spike trace
    game  <--actions----  \\\\.\\pipe\\botw_fly_bridge  <--  ActionDecoder  <--  linear readout  <--+

The brain is frozen (fly.ai's FlyBrain, 166,700 neurons, 20 ms steps).  Only the linear readout is
trained, from outcomes the game reports (parry, block, dodge, hit taken ...), in hindsight.

    python main.py check                  # does Link's world reach the fly's neurons? how fast is a brain step?
    python main.py train                  # practise against real enemies; the fly also tries random things
    python main.py play                   # the fly fights; it keeps learning (--no-learn freezes it)
    python main.py fit                    # refit the readout from saved experience and print the report
    python main.py sim                    # whole loop against a mock arena and a stand-in brain (no game, no download)

Action packet we send back (one per telemetry packet; same framing; --codec binary is 18 bytes, "BFLA" + flags + seq + 2 floats):
    {"v":1,"seq":812,"move":[0.40,-0.20],"shield":true,"parry":false,"jump":false,"attack":false}
move is Link's stick (+x right, +z forward); shield is a level; parry/jump/attack are one-shot (True for exactly one packet).

Useful flags:  --pipe \\\\.\\pipe\\botw_fly_bridge   (or tcp://127.0.0.1:5555 / unix:///tmp/botw.sock for development)
               --framing len|line|message   how the C++ side delimits packets (default: 4-byte length prefix)
               --codec json|binary          format of the ACTION packets we send (telemetry is auto-detected)
               --device auto|cpu|cuda       FlyBrain device (auto = GPU if there is one)
               --encoder loom_size=0.5,amp_lc4=0.9    override EncoderConfig fields (also --label --readout --decoder --explorer)
               --steps auto|1|2|3           brain steps per packet (auto = keep brain time close to game time)
               --fresh                      ignore saved experience; --memory-dir botw_memory
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import select
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from botw_encoder import (AudioCue, AudioKind, BotwEncoder, CellMap, EncoderConfig, Entity, EntityKind, EventKind,
                          OutcomeInference, Telemetry, decode_packet, merge_packets, parse_overrides)
from botw_readout import (HEADS, ActionDecoder, ActionState, AsyncFitter, DecoderConfig, ExperienceMemory,
                          ExploreConfig, Explorer, HindsightLabeller, LabelConfig, LinearReadout, ReadoutConfig,
                          SpikeTrace, TickRecord)

DEFAULT_PIPE = r"\\.\pipe\botw_fly_bridge"
DEFENSIVE = (EventKind.HIT_TAKEN, EventKind.PARRY, EventKind.BLOCK, EventKind.PERFECT_DODGE, EventKind.DODGE)


class EMA:
    def __init__(self, a: float = 0.1):
        self.a, self.value, self.n = a, 0.0, 0

    def update(self, x: float) -> None:
        self.value = x if self.n == 0 else (1 - self.a) * self.value + self.a * x
        self.n += 1


# ====================================================================== transports
class Transport:
    def read(self, timeout: float) -> bytes:
        """Whatever bytes are available (waiting up to ``timeout`` s for the first); b"" on timeout.
        Raises ConnectionError when the other side is gone."""
        raise NotImplementedError

    def write(self, data: bytes) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SocketTransport(Transport):
    def __init__(self, sock: socket.socket):
        self.sock = sock

    def read(self, timeout: float) -> bytes:
        try:
            ready, _, _ = select.select([self.sock], [], [], max(timeout, 0.0))
            if not ready:
                return b""
            data = self.sock.recv(65536)
        except (OSError, ValueError) as exc:
            raise ConnectionError(str(exc)) from exc
        if not data:
            raise ConnectionError("the bridge closed the connection")
        return data

    def write(self, data: bytes) -> None:
        try:
            self.sock.sendall(data)
        except OSError as exc:
            raise ConnectionError(str(exc)) from exc

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class NamedPipeTransport(Transport):
    """Client end of a Windows named pipe, straight through kernel32 (no pywin32 needed).

    Waiting is done by polling PeekNamedPipe, so a read never blocks past its timeout and the game
    loop stays responsive.  ``message_mode`` switches the pipe to message read mode: use it when
    the C++ server was created with PIPE_TYPE_MESSAGE and sends one message per packet."""
    GENERIC_READ, GENERIC_WRITE, OPEN_EXISTING = 0x80000000, 0x40000000, 3
    ERROR_FILE_NOT_FOUND, ERROR_PIPE_BUSY, ERROR_MORE_DATA = 2, 231, 234
    PIPE_READMODE_MESSAGE = 2
    BUF = 65536
    _require_windows = True

    @staticmethod
    def _load():
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        P = ctypes.POINTER
        k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                  wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        k.CreateFileW.restype = wintypes.HANDLE
        k.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, P(wintypes.DWORD), wintypes.LPVOID]
        k.ReadFile.restype = wintypes.BOOL
        k.WriteFile.argtypes = [wintypes.HANDLE, wintypes.LPCVOID, wintypes.DWORD, P(wintypes.DWORD), wintypes.LPVOID]
        k.WriteFile.restype = wintypes.BOOL
        k.PeekNamedPipe.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, P(wintypes.DWORD),
                                    P(wintypes.DWORD), P(wintypes.DWORD)]
        k.PeekNamedPipe.restype = wintypes.BOOL
        k.SetNamedPipeHandleState.argtypes = [wintypes.HANDLE, P(wintypes.DWORD), P(wintypes.DWORD), P(wintypes.DWORD)]
        k.SetNamedPipeHandleState.restype = wintypes.BOOL
        k.WaitNamedPipeW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD]
        k.WaitNamedPipeW.restype = wintypes.BOOL
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.CloseHandle.restype = wintypes.BOOL
        return k

    @staticmethod
    def _last_error() -> int:
        import ctypes
        return ctypes.get_last_error()

    def __init__(self, name: str, message_mode: bool = False, wait: Optional[float] = None, log=print):
        if self._require_windows and os.name != "nt":
            raise OSError("named pipes need Windows.  Elsewhere use --pipe tcp://127.0.0.1:PORT or unix:///path")
        import ctypes
        from ctypes import wintypes
        self._ct, self._dw = ctypes, wintypes.DWORD
        self.k = self._load()
        self.name = name
        invalid = ctypes.c_void_p(-1).value
        deadline = None if wait is None else time.monotonic() + wait
        warned = 0.0
        while True:
            h = self.k.CreateFileW(name, self.GENERIC_READ | self.GENERIC_WRITE, 0, None, self.OPEN_EXISTING, 0, None)
            if h not in (None, invalid):
                self.h = h
                break
            err = self._last_error()
            if err == self.ERROR_PIPE_BUSY:                       # server has it, another client is attached
                self.k.WaitNamedPipeW(name, 1000)
                continue
            if err != self.ERROR_FILE_NOT_FOUND:
                raise ConnectionError(f"cannot open {name} (Windows error {err})")
            if deadline is not None and time.monotonic() > deadline:
                raise ConnectionError(f"{name} does not exist: is the game running with the bridge loaded?")
            if time.monotonic() - warned > 5.0:
                log(f"waiting for {name} ...")
                warned = time.monotonic()
            time.sleep(0.25)
        self._timer = False
        if os.name == "nt":                                       # 1 ms sleeps (the default tick is ~15 ms before Python 3.11)
            try:
                ctypes.WinDLL("winmm").timeBeginPeriod(1)
                self._timer = True
            except OSError:
                pass
        if message_mode:
            mode = wintypes.DWORD(self.PIPE_READMODE_MESSAGE)
            if not self.k.SetNamedPipeHandleState(self.h, ctypes.byref(mode), None, None):
                raise ConnectionError(f"could not switch {name} to message mode (error {self._last_error()})")

    def read(self, timeout: float) -> bytes:
        ct, k = self._ct, self.k
        start = time.monotonic()
        deadline = start + max(timeout, 0.0)
        avail = self._dw(0)
        while True:
            if not k.PeekNamedPipe(self.h, None, 0, None, ct.byref(avail), None):
                raise ConnectionError(f"pipe closed (error {self._last_error()})")
            if avail.value:
                break
            now = time.monotonic()
            if now >= deadline:
                return b""
            if now - start > 0.002:                                # spin for the first 2 ms, then sleep between polls
                time.sleep(0.001)
        buf = ct.create_string_buffer(self.BUF)
        got = self._dw(0)
        chunks = []
        while True:
            ok = k.ReadFile(self.h, buf, self.BUF, ct.byref(got), None)
            chunks.append(buf.raw[:got.value])
            if ok:
                break
            err = self._last_error()
            if err != self.ERROR_MORE_DATA:                        # more of the same message: keep reading
                raise ConnectionError(f"pipe read failed (error {err})")
        return b"".join(chunks)

    def write(self, data: bytes) -> None:
        n = self._dw(0)
        if not self.k.WriteFile(self.h, data, len(data), self._ct.byref(n), None):
            raise ConnectionError(f"pipe write failed (error {self._last_error()})")

    def close(self) -> None:
        if getattr(self, "h", None) is not None:
            self.k.CloseHandle(self.h)
            self.h = None
        if getattr(self, "_timer", False):
            try:
                self._ct.WinDLL("winmm").timeEndPeriod(1)
            except OSError:
                pass
            self._timer = False


def open_transport(spec: str, message_mode: bool = False, wait: Optional[float] = None, log=print) -> Transport:
    """\\\\.\\pipe\\name (Windows), tcp://host:port or unix:///path (development on other systems / WSL)."""
    if spec.startswith("tcp://"):
        host, _, port = spec[6:].rpartition(":")
        family, addr = socket.AF_INET, (host or "127.0.0.1", int(port))
    elif spec.startswith("unix://"):
        family, addr = socket.AF_UNIX, spec[7:]
    else:
        return NamedPipeTransport(spec, message_mode, wait, log)
    deadline = None if wait is None else time.monotonic() + wait
    while True:
        s = socket.socket(family, socket.SOCK_STREAM)
        try:
            s.connect(addr)
            if family == socket.AF_INET:
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return SocketTransport(s)
        except OSError as exc:
            s.close()
            if deadline is not None and time.monotonic() > deadline:
                raise ConnectionError(f"cannot reach {spec}: {exc}") from exc
            log(f"waiting for {spec} ...")
            time.sleep(1.0)


# ====================================================================== framing and client
class Framer:
    """len: 4-byte little-endian length prefix.  line: one packet per line (JSON only).
    message: every read is one packet (pipe in message mode)."""
    MAX = 1 << 20

    def __init__(self, mode: str = "len"):
        if mode not in ("len", "line", "message"):
            raise ValueError(f"unknown framing {mode!r}")
        self.mode, self.buf = mode, bytearray()

    def feed(self, data: bytes) -> List[bytes]:
        if self.mode == "message":
            return [bytes(data)] if data else []
        self.buf += data
        out: List[bytes] = []
        if self.mode == "line":
            while True:
                i = self.buf.find(b"\n")
                if i < 0:
                    break
                line = bytes(self.buf[:i]).rstrip(b"\r")
                del self.buf[:i + 1]
                if line:
                    out.append(line)
            if len(self.buf) > self.MAX:
                raise ValueError("no newline in 1 MB of data: wrong --framing?")
            return out
        while len(self.buf) >= 4:
            (n,) = struct.unpack_from("<I", self.buf, 0)
            if n > self.MAX:
                raise ValueError(f"packet length {n} is absurd: wrong --framing? (the C++ side may send lines or pipe messages)")
            if len(self.buf) < 4 + n:
                break
            out.append(bytes(self.buf[4:4 + n]))
            del self.buf[:4 + n]
        return out

    def pack(self, payload: bytes) -> bytes:
        if self.mode == "len":
            return struct.pack("<I", len(payload)) + payload
        if self.mode == "line":
            return payload + b"\n"
        return payload


class BridgeClient:
    def __init__(self, transport: Transport, framing: str = "len", codec: str = "json"):
        if framing == "line" and codec == "binary":
            raise ValueError("line framing cannot carry binary packets; use --framing len or message")
        self.transport, self.framer, self.codec = transport, Framer(framing), codec
        self.dropped = 0
        self.bad = 0

    def recv_latest(self, timeout: float = 0.5) -> Optional[Telemetry]:
        """The newest telemetry.  If packets piled up while the brain was busy, only the newest state is
        used, but every event in the backlog is kept and the elapsed time is summed."""
        data = self.transport.read(timeout)
        if not data:
            return None
        frames = self.framer.feed(data)
        while True:
            more = self.transport.read(0.0)
            if not more:
                break
            frames += self.framer.feed(more)
        packets = []
        for f in frames:
            try:
                packets.append(decode_packet(f))
            except (ValueError, struct.error) as exc:
                self.bad += 1
                if self.bad <= 3:
                    print(f"[bridge] bad telemetry packet ignored: {exc}")
        if not packets:
            return None
        self.dropped += len(packets) - 1
        return merge_packets(packets)

    def send(self, act: ActionState) -> None:
        payload = act.to_binary() if self.codec == "binary" else act.to_json()
        self.transport.write(self.framer.pack(payload))

    def close(self) -> None:
        self.transport.close()


# ====================================================================== the agent
@dataclass
class AgentConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    explore: ExploreConfig = field(default_factory=ExploreConfig)
    trace_tau: float = 0.1
    max_rows: int = 30000
    refit_every: int = 10          # outcomes between refits
    steps: str = "auto"            # brain steps per packet: "auto" or an integer
    max_steps: int = 3
    seed: int = 0
    sync_fit: bool = False
    audio_types: Optional[tuple] = None


class FlyAgent:
    """Encoder -> FlyBrain -> spike trace -> linear readout -> action, plus the hindsight learning loop."""

    def __init__(self, brain, cfg: Optional[AgentConfig] = None, log=print):
        self.brain, self.cfg, self.log = brain, cfg or AgentConfig(), log
        self.cells = CellMap(brain, self.cfg.audio_types)
        self.encoder = BotwEncoder(self.cells, self.cfg.encoder, self.cfg.seed)
        self.trace = SpikeTrace(self.cells.dn, brain.dt, self.cfg.trace_tau)
        self.memory = ExperienceMemory(self.trace.n, self.cfg.max_rows)
        self.labeller = HindsightLabeller(self.memory, self.cfg.label)
        self.decoder = ActionDecoder(self.cfg.decoder)
        self.explorer = Explorer(self.cfg.explore, self.cfg.seed)
        self.readout: Optional[LinearReadout] = None
        self.fitter = AsyncFitter(self.memory, self.cfg.readout, self._set_readout)
        self.infer = OutcomeInference()
        self.ms = EMA(0.1)
        self.last_steps = 1
        self.last_t = 0.0
        self.outcomes: collections.Counter = collections.Counter()
        self.fits = 0

    # ------------------------------------------------------------------ one packet
    def _substeps(self, dt: float) -> int:
        c = self.cfg
        want = int(round(dt / self.brain.dt)) if c.steps == "auto" else int(c.steps)
        want = max(1, min(want, c.max_steps))
        if self.ms.n and dt > 0:                       # latency guard: never plan more brain time than the packet allows
            want = max(1, min(want, int(0.8 * dt * 1000.0 / max(self.ms.value, 1e-3))))
        return want

    def sense(self, tel: Telemetry):
        """Encode a packet, step the brain, update the spike trace.  Returns (percept, features)."""
        percept = self.encoder.encode(tel)
        dt = tel.dt if tel.dt > 1e-4 else 1.0 / 30.0
        n = self._substeps(dt)
        t0 = time.perf_counter()
        for _ in range(n):
            self.trace.update(self.brain.step(inject=percept.inject))
        self.ms.update((time.perf_counter() - t0) * 1000.0 / n)
        self.last_steps = n
        return percept, self.trace.vector()

    def tick(self, tel: Telemetry, learn: bool = True, explore: bool = False) -> ActionState:
        percept, x = self.sense(tel)
        self.last_t = tel.t
        ro = self.readout
        scores = ro.scores(x) if ro is not None else None
        act = self.decoder.decide(scores, ro, tel.link, tel.t)
        if explore:
            trial = self.explorer.step(tel.t, tel.dt, percept, sum(self.outcomes[k] for k in DEFENSIVE))
            if trial is not None:
                trial.seq = act.seq
                act = trial
        events = self.infer(tel)
        for ev in events:
            self.outcomes[EventKind(ev.kind)] += 1
        if learn:
            self.labeller.push(TickRecord(t=tel.t, x=x, a=act.vector(), threat=percept.threat,
                                          bearing=percept.bearing, evade=percept.evade))
            for ev in events:
                self.labeller.resolve(ev, tel.link.hp_max)
            self._maybe_refit()
        return act

    # ------------------------------------------------------------------ learning
    def _set_readout(self, ro: LinearReadout) -> None:
        self.readout = ro
        self.fits += 1

    def _maybe_refit(self) -> None:
        if (self.labeller.outcomes_since_fit >= self.cfg.refit_every and self.memory.n >= self.cfg.readout.min_rows
                and not self.fitter.busy):
            self.labeller.outcomes_since_fit = 0
            self.fitter.submit(blocking=self.cfg.sync_fit)

    def refit_now(self) -> Optional[LinearReadout]:
        while self.fitter.busy:
            time.sleep(0.05)
        self.fitter.submit(blocking=True)
        if self.fitter.error:
            self.log(f"[fit] failed: {self.fitter.error!r}")
        return self.readout

    def reset_session(self) -> None:
        """A new connection (or the game went away): forget the live state, keep what was learned."""
        self.trace.reset()
        self.labeller.pending.clear()
        self.decoder = ActionDecoder(self.cfg.decoder)
        self.explorer = Explorer(self.cfg.explore, self.cfg.seed)
        self.infer = OutcomeInference()
        self.encoder._prev_d.clear()
        self.encoder._prev_closing.clear()

    # ------------------------------------------------------------------ persistence
    def save(self, d: str) -> None:
        os.makedirs(d, exist_ok=True)

        def atomic(name, writer):
            final = os.path.join(d, name)
            tmp = final[:-4] + ".tmp.npz"
            writer(tmp)
            os.replace(tmp, final)

        atomic("memory.npz", self.memory.save)
        if self.readout is not None:
            atomic("readout.npz", self.readout.save)
        with open(os.path.join(d, "stats.json"), "w") as fh:
            json.dump({"outcomes": {k.name: v for k, v in self.outcomes.items()}, "rows": self.memory.n,
                       "fits": self.fits}, fh, indent=1)

    def load(self, d: str) -> str:
        msgs = []
        mp, rp = os.path.join(d, "memory.npz"), os.path.join(d, "readout.npz")
        if os.path.exists(mp):
            try:
                self.memory.adopt(ExperienceMemory.load(mp, self.trace.n, self.cfg.max_rows))
                msgs.append(f"{self.memory.n} labelled ticks")
            except ValueError as exc:
                self.log(f"[memory] ignoring saved experience: {exc}")
        if os.path.exists(rp):
            try:
                ro = LinearReadout.load(rp)
                if ro.W.shape[0] == self.trace.n:
                    self.readout = ro
                    msgs.append("readout (%s useful)" % (",".join(h for h, u in zip(HEADS, ro.useful) if u) or "no head"))
            except Exception as exc:
                self.log(f"[readout] ignoring saved readout: {exc}")
        return ", ".join(msgs)

    def hud(self, label: str, pkts: int, secs: float, client: Optional[BridgeClient] = None) -> str:
        o = self.outcomes
        useful = ",".join(h for h, u in zip(HEADS, self.readout.useful) if u) if self.readout else "-"
        eps = max(self.cfg.explore.min_scale, 1.0 - sum(o[k] for k in DEFENSIVE) / max(self.cfg.explore.decay_outcomes, 1))
        return (f"[{label}] t={self.last_t:7.1f}s {pkts / max(secs, 1e-6):5.1f} pkt/s | brain {self.ms.value:5.1f} ms x{self.last_steps}"
                f" | parry {o[EventKind.PARRY]} block {o[EventKind.BLOCK]} dodge {o[EventKind.DODGE] + o[EventKind.PERFECT_DODGE]}"
                f" hit {o[EventKind.HIT_TAKEN]} landed {o[EventKind.HIT_LANDED]} | rows {self.memory.n} fits {self.fits}"
                f" useful[{useful}] eps {eps:.2f}" + (f" drop {client.dropped}" if client and client.dropped else ""))

    @property
    def defensive_total(self) -> int:
        return sum(self.outcomes[k] for k in DEFENSIVE)


# ====================================================================== setup helpers
def make_brain(args):
    if args.fake_brain:
        from botw_sim import StandInBrain
        return StandInBrain(seed=args.seed)
    try:
        from flybrain import FlyBrain
    except ImportError:
        raise SystemExit("flybrain is not installed:  pip install flybrain   (GPU: pip install \"flybrain[gpu]\", then --device cuda)\n"
                         "To test the loop without it:  python main.py sim")
    try:
        return FlyBrain(device=args.device, seed=args.seed)
    except TypeError:                                  # an older build without the seed option
        return FlyBrain(device=args.device)


def make_config(args) -> AgentConfig:
    cfg = AgentConfig(seed=args.seed, steps=str(args.steps), max_steps=args.max_steps, sync_fit=getattr(args, "sync_fit", False),
                      refit_every=args.refit_every, max_rows=args.max_rows, trace_tau=args.trace_tau,
                      audio_types=tuple(args.audio_types.split(",")) if args.audio_types else None)
    parse_overrides(cfg.encoder, args.encoder)
    parse_overrides(cfg.label, args.label)
    parse_overrides(cfg.readout, args.readout)
    parse_overrides(cfg.decoder, args.decoder)
    parse_overrides(cfg.explore, args.explorer)
    return cfg


def build_agent(args, load: bool = True) -> FlyAgent:
    print("loading the connectome ..." if not args.fake_brain else "using the stand-in brain ...", flush=True)
    t0 = time.time()
    brain = make_brain(args)
    agent = FlyAgent(brain, make_config(args))
    print(f"brain ready in {time.time() - t0:.1f}s; {agent.trace.n} descending neurons feed the readout", flush=True)
    for problem in agent.cells.problems():
        print(f"  WARNING: {problem}")
    if load and not args.fresh:
        got = agent.load(args.memory_dir)
        if got:
            print(f"loaded from {args.memory_dir}: {got}")
    return agent


def run_loop(agent: FlyAgent, client: BridgeClient, *, learn: bool, explore: bool, label: str,
             max_outcomes: Optional[int] = None, max_seconds: Optional[float] = None, hud_every: float = 2.0,
             quiet: bool = False, stop: Optional[threading.Event] = None) -> int:
    t_start = last_hud = time.time()
    n = last_n = 0
    while stop is None or not stop.is_set():
        tel = client.recv_latest(timeout=0.5)
        now = time.time()
        if tel is None:
            if max_seconds and now - t_start > max_seconds:
                break
            if not quiet and now - last_hud > 5.0:
                print(f"[{label}] no telemetry yet ...", flush=True)
                last_hud = now
            continue
        client.send(agent.tick(tel, learn=learn, explore=explore))
        n += 1
        if max_outcomes and agent.defensive_total >= max_outcomes:
            break
        if max_seconds and now - t_start > max_seconds:
            break
        if not quiet and now - last_hud >= hud_every:
            print(agent.hud(label, n - last_n, now - last_hud, client), flush=True)
            last_hud, last_n = now, n
    return n


def live_session(args, agent: FlyAgent, *, learn: bool, explore: bool, label: str, max_outcomes: Optional[int]) -> None:
    """Connect to the game and run until Ctrl-C or the outcome budget; reconnects if the game restarts."""
    try:
        while True:
            try:
                transport = open_transport(args.pipe, args.framing == "message", wait=args.connect_timeout)
            except ConnectionError as exc:
                print(exc)
                return
            client = BridgeClient(transport, args.framing, args.codec)
            agent.reset_session()
            print(f"connected to {args.pipe}  (framing={args.framing}, actions={args.codec})", flush=True)
            try:
                run_loop(agent, client, learn=learn, explore=explore, label=label, max_outcomes=max_outcomes,
                         hud_every=args.hud_every)
                return
            except ConnectionError as exc:
                print(f"[bridge] connection lost ({exc}); saving and waiting for the game ...")
                agent.save(args.memory_dir)
                agent.reset_session()
                time.sleep(1.0)
            except ValueError as exc:
                print(f"[bridge] {exc}")
                return
            finally:
                client.close()
    finally:
        if learn:
            print("final fit ...", flush=True)
            if agent.memory.n >= agent.cfg.readout.min_rows:
                agent.refit_now()
            agent.save(args.memory_dir)
            print(f"saved to {args.memory_dir}")
        if agent.readout is not None:
            print(agent.readout.format_report())


# ====================================================================== commands
def scenario_telemetry():
    def enemy(x, z):
        return Telemetry(entities=[Entity(id=1, kind=EntityKind.ENEMY, pos=(x, 0.0, z), radius=0.6, windup=0.8, attacking=True)])

    def arrow(x, z):
        p = np.array([x, 1.0, z])
        v = -p / np.linalg.norm(p) * 20.0
        return Telemetry(entities=[Entity(id=2, kind=EntityKind.PROJECTILE, pos=tuple(p), vel=tuple(v), radius=0.12)])

    def sound(az):
        return Telemetry(audio=[AudioCue(kind=AudioKind.CHARGE, azimuth=az, level=1.0)])

    return [("idle", Telemetry()), ("enemy left", enemy(-3, 3)), ("enemy right", enemy(3, 3)), ("enemy ahead", enemy(0, 4)),
            ("arrow left", arrow(-5, 12)), ("arrow right", arrow(5, 12)), ("sound left", sound(-1.3)), ("sound right", sound(1.3))]


def cmd_check(args) -> int:
    agent = build_agent(args, load=False)
    print("\nneurons the court can drive:")
    print(agent.cells.describe())
    pos = lambda idx: np.searchsorted(agent.trace.idx, idx[np.isin(idx, agent.trace.idx)])
    sets = {"L-DNs": agent.cells.dn_side["L"], "R-DNs": agent.cells.dn_side["R"]}
    for name in ("DNa02", "DNp01"):
        for s in "LR":
            sets[f"{name}.{s}"] = agent.cells.named[name][s]
    for name in ("DNg100", "MDN"):
        d = agent.cells.named[name]
        sets[name] = np.unique(np.concatenate([d[s] for s in d]))

    # brain step time with a realistic input
    probe = Telemetry(dt=1 / 30, entities=[Entity(id=1, kind=EntityKind.ENEMY, pos=(-2.0, 0.0, 3.0), radius=0.6, windup=0.7)])
    inject = agent.encoder.encode(probe).inject
    t0 = time.perf_counter()
    agent.brain.step(inject=inject)
    print(f"\nfirst brain step: {time.perf_counter() - t0:.2f}s (the first call compiles numba)")
    times = []
    for _ in range(args.ticks + 10):
        t0 = time.perf_counter()
        agent.trace.update(agent.brain.step(inject=inject))
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times[10:])
    print(f"brain step: median {np.median(times):.1f} ms, 95th percentile {np.percentile(times, 95):.1f} ms "
          f"(simulated step = {agent.brain.dt * 1000:.0f} ms; a 30 FPS game gives you 33 ms per frame)")

    # scenarios
    results = {}
    for name, tel in scenario_telemetry():
        tel.dt = 1 / 30
        for _ in range(25):                                   # let the previous scenario die away
            agent.trace.update(agent.brain.step(inject=[]))
        agent.trace.reset()
        percept = agent.encoder.encode(tel)
        frames = []
        for _ in range(args.ticks):
            agent.trace.update(agent.brain.step(inject=percept.inject))
            frames.append(agent.trace.vector())
        results[name] = np.mean(frames[len(frames) // 3:], axis=0)
    idle = results["idle"]
    cols = list(sets)
    print("\nmean descending-neuron activity (decaying spike count, last 100 ms) per scenario:")
    print(f"{'':<12}" + "".join(f"{c:>9}" for c in cols) + f"{'DNs>+0.25':>11}")
    for name, x in results.items():
        row = "".join(f"{x[pos(sets[c])].sum():>9.1f}" for c in cols)
        print(f"{name:<12}{row}{int(((x - idle) > 0.25).sum()):>11}")

    def pattern_distance(a, b):
        da, db = results[a] - idle, results[b] - idle
        den = float(np.linalg.norm(da) * np.linalg.norm(db))
        return 1.0 - float(da @ db) / den if den > 1e-9 else float("nan")

    ok = True
    print()
    for a, b in (("enemy left", "enemy right"), ("arrow left", "arrow right"), ("sound left", "sound right")):
        d = pattern_distance(a, b)
        flag = "ok" if d > 0.2 else "WEAK"
        ok &= flag == "ok"
        print(f"  {a} vs {b}: pattern distance {d:.2f}  [{flag}]")
    moved = sum(int(((results[n] - idle) > 0.25).sum()) for n in ("enemy left", "enemy right", "enemy ahead"))
    if moved == 0:
        print("\nNothing responds to the enemy scenarios: raise the amplitudes, e.g. --encoder amp_loom=1.2,amp_threat=1.2")
    elif not ok:
        print("\nSome left/right pairs look alike.  The README's advice: raise the AMP values (--encoder amp_loom=...) "
              "or check that the cell types above exist on both sides.")
    else:
        print("\nLeft and right scenarios drive different neurons.")
    return 0


def cmd_train(args) -> int:
    agent = build_agent(args)
    live_session(args, agent, learn=True, explore=not args.no_explore, label="train", max_outcomes=args.incidents)
    return 0


def cmd_play(args) -> int:
    agent = build_agent(args)
    if agent.readout is None:
        print("note: no trained readout yet, so the fly will stand still until it has learned something "
              "(run `train` first, or play with learning on and let it get hit).")
    live_session(args, agent, learn=not args.no_learn, explore=False, label="play", max_outcomes=None)
    return 0


def cmd_fit(args) -> int:
    agent = build_agent(args)
    if agent.memory.n < agent.cfg.readout.min_rows:
        print(f"only {agent.memory.n} labelled ticks saved in {args.memory_dir}; need {agent.cfg.readout.min_rows}")
        return 1
    ro = agent.refit_now()
    print(ro.format_report())
    agent.save(args.memory_dir)
    return 0


def run_mock_session(agent: FlyAgent, mock, args, *, learn: bool, explore: bool, label: str, max_outcomes: int,
                     max_seconds: float = 600.0, quiet: bool = True) -> None:
    a, b = socket.socketpair()
    server_io, stop = SocketTransport(a), threading.Event()
    framer, backlog = Framer(args.framing), collections.deque()

    def send_frame(payload: bytes) -> None:
        server_io.write(framer.pack(payload))

    def recv_frame() -> Optional[bytes]:
        while not backlog:
            try:
                data = server_io.read(0.2)
            except ConnectionError:
                stop.set()
                return None
            if stop.is_set():
                return None
            if data:
                backlog.extend(framer.feed(data))
            else:
                return None
        return backlog.popleft()

    def server():
        try:
            mock.serve(send_frame, recv_frame, stop.is_set)
        except ConnectionError:
            pass

    agent.reset_session()
    th = threading.Thread(target=server, daemon=True)
    th.start()
    client = BridgeClient(SocketTransport(b), args.framing, args.codec)
    try:
        run_loop(agent, client, learn=learn, explore=explore, label=label, max_outcomes=max_outcomes,
                 max_seconds=max_seconds, hud_every=args.hud_every, quiet=quiet)
    finally:
        stop.set()
        client.close()
        th.join(timeout=2.0)
        server_io.close()


def cmd_sim(args) -> int:
    from botw_sim import MockBotw
    args.fake_brain = args.fake_brain or not args.real_brain
    args.sync_fit = True
    args.fresh = True
    agent = build_agent(args, load=False)
    print("\nThis checks the plumbing with a stand-in brain; it says nothing about how well the real fly would play.\n")

    def outcome_line(mock: MockBotw) -> str:
        o = mock.outcomes
        d = [o[k.name] for k in DEFENSIVE]
        total = max(sum(d), 1)
        return (f"hit {o['HIT_TAKEN']:>3} ({100 * o['HIT_TAKEN'] / total:4.1f}%)  parry {o['PARRY']:>3}  block {o['BLOCK']:>3}  "
                f"perfect dodge {o['PERFECT_DODGE']:>3}  sidestep {o['DODGE']:>3}  hits landed {o['HIT_LANDED']:>3}")

    print(f"baseline: an untrained fly for {args.eval_outcomes} threats ...", flush=True)
    base = MockBotw(seed=args.seed + 1000, codec=args.codec)
    run_mock_session(agent, base, args, learn=False, explore=False, label="baseline", max_outcomes=args.eval_outcomes)
    print("  " + outcome_line(base))
    agent.outcomes.clear()

    print(f"\ntraining: {args.incidents} outcomes with exploration and hindsight labels ...", flush=True)
    t0 = time.time()
    train = MockBotw(seed=args.seed, codec=args.codec)
    run_mock_session(agent, train, args, learn=True, explore=True, label="train", max_outcomes=args.incidents,
                     quiet=False)
    agent.refit_now()
    print(f"  {outcome_line(train)}   ({time.time() - t0:.0f}s, {agent.memory.n} labelled ticks)\n")
    print(agent.readout.format_report() if agent.readout else "no readout was fitted")

    agent.outcomes.clear()
    print(f"\nevaluation: the trained fly, learning off, same {args.eval_outcomes} threats as the baseline ...", flush=True)
    test = MockBotw(seed=args.seed + 1000, codec=args.codec)
    run_mock_session(agent, test, args, learn=False, explore=False, label="eval", max_outcomes=args.eval_outcomes)
    print("  " + outcome_line(test))
    b_hit, t_hit = base.outcomes["HIT_TAKEN"], test.outcomes["HIT_TAKEN"]
    b_def = max(sum(base.outcomes[k.name] for k in DEFENSIVE), 1)
    t_def = max(sum(test.outcomes[k.name] for k in DEFENSIVE), 1)
    print(f"\nhit rate: {100 * b_hit / b_def:.0f}% untrained -> {100 * t_hit / t_def:.0f}% trained")
    return 0


# ====================================================================== command line
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pipe", default=DEFAULT_PIPE, help=r"named pipe (\\.\pipe\name), tcp://host:port or unix:///path")
    common.add_argument("--framing", choices=("len", "line", "message"), default="len")
    common.add_argument("--codec", choices=("json", "binary"), default="json", help="format of the action packets we send")
    common.add_argument("--connect-timeout", type=float, default=None, help="seconds to wait for the bridge (default: forever)")
    common.add_argument("--device", default=os.environ.get("FLY_DEVICE", "auto"), help="auto, cpu or cuda")
    common.add_argument("--seed", type=int, default=1)
    common.add_argument("--steps", default="auto", help="brain steps per packet: auto, 1, 2 or 3")
    common.add_argument("--max-steps", type=int, default=3, help="upper limit for --steps auto")
    common.add_argument("--memory-dir", default="botw_memory")
    common.add_argument("--fresh", action="store_true", help="ignore saved experience and readout")
    common.add_argument("--fake-brain", action="store_true", help="use the stand-in brain from botw_sim.py")
    common.add_argument("--encoder", default="", help="EncoderConfig overrides: loom_size=0.5,amp_lc4=0.9")
    common.add_argument("--label", default="", help="LabelConfig overrides")
    common.add_argument("--readout", default="", help="ReadoutConfig overrides (tuples with ':' : ranks=5:20)")
    common.add_argument("--decoder", default="", help="DecoderConfig overrides")
    common.add_argument("--explorer", default="", help="ExploreConfig overrides")
    common.add_argument("--audio-types", default="", help="comma list of hearing cell types, e.g. JO-A,JO-B")
    common.add_argument("--refit-every", type=int, default=10, help="outcomes between readout refits")
    common.add_argument("--max-rows", type=int, default=30000)
    common.add_argument("--trace-tau", type=float, default=0.1)
    common.add_argument("--hud-every", type=float, default=2.0)

    ap = argparse.ArgumentParser(prog="main.py", description="A fruit-fly connectome plays Breath of the Wild.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", parents=[common], help="diagnose the encoder and time the brain (no game needed)")
    c.add_argument("--ticks", type=int, default=40)
    t = sub.add_parser("train", parents=[common], help="learn from outcomes with random exploration")
    t.add_argument("--incidents", type=int, default=300, help="stop after this many defensive outcomes")
    t.add_argument("--no-explore", action="store_true")
    p = sub.add_parser("play", parents=[common], help="fight; keeps learning unless --no-learn")
    p.add_argument("--no-learn", action="store_true")
    sub.add_parser("fit", parents=[common], help="refit the readout from saved experience")
    s = sub.add_parser("sim", parents=[common], help="whole loop against a mock arena (no game, no download)")
    s.add_argument("--incidents", type=int, default=250)
    s.add_argument("--eval-outcomes", type=int, default=80)
    s.add_argument("--real-brain", action="store_true", help="use flybrain instead of the stand-in")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if getattr(args, "framing", "") == "line" and getattr(args, "codec", "") == "binary":
        ap.error("--framing line cannot carry --codec binary (binary data can contain newlines); use --framing len or message")
    handlers = {"check": cmd_check, "train": cmd_train, "play": cmd_play, "fit": cmd_fit, "sim": cmd_sim}
    try:
        make_config(args)                                 # fail early, with a readable message, on a bad --encoder/--label/... string
    except ValueError as exc:
        ap.error(str(exc))
    try:
        return handlers[args.cmd](args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
