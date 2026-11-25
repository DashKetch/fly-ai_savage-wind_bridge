"""botw_sim.py -- offline test harness: a stand-in brain and a mock Breath of the Wild arena.

Neither is the real thing.  They exist so the whole loop (pipe client, encoder, brain stepping,
hindsight labelling, readout fit, action decoding) can be run and checked without the game and
without downloading the connectome:  ``python main.py sim``.

StandInBrain  a small random spiking network exposing only the calls this project uses:
              ``cells(types, side=None)``, ``step(inject=...)`` and ``dt``.  It is NOT the fly:
              results with it say that the plumbing works, nothing about how the real fly plays.
MockBotw      an arena where melee enemies and arrows come at Link and are resolved with simple
              rules (parry window, guard, dodge window, sidestep).  It speaks the real telemetry and
              action formats, including outcome events.
"""
from __future__ import annotations

import collections
import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from botw_encoder import (AudioCue, AudioKind, Entity, EntityKind, EventKind, GameEvent, LinkState, Telemetry,
                          encode_binary, encode_json)
from botw_readout import ActionState, decode_action


# ====================================================================== stand-in brain
class StandInBrain:
    dt = 0.02

    def __init__(self, seed: int = 0, n_hidden: int = 800, n_dn: int = 240, pop: int = 40,
                 g_in: float = -3.6, g_out: float = -3.4):
        rng = np.random.default_rng(seed)
        self._sets: Dict[tuple, np.ndarray] = {}
        p = 0
        for t in ("LPLC2", "LC4", "LPLC1", "LC10a"):
            for s in "LR":
                self._sets[(t, s)] = np.arange(p, p + pop)
                p += pop
        for s in "LR":
            self._sets[("JO-A", s)] = np.arange(p, p + pop // 2)
            p += pop // 2
        hid = np.arange(p, p + n_hidden)
        p += n_hidden
        dn = np.arange(p, p + n_dn)
        p += n_dn
        self.n = p
        self._hid = {"L": hid[: n_hidden // 2], "R": hid[n_hidden // 2:]}
        self._dn_side = {"L": dn[: n_dn // 2], "R": dn[n_dn // 2:]}
        self._sets[("descending_neuron", None)] = dn
        for s in "LR":
            self._sets[("descending_neuron", s)] = self._dn_side[s]
        # a few identified "command" neurons, so `check` has something to show
        for i, name in enumerate(("DNa02", "DNp01", "DNg100", "MDN")):
            for s in "LR":
                self._sets[(name, s)] = self._dn_side[s][i * 3: i * 3 + 3]

        W = np.zeros((p, p), np.float32)
        other = {"L": "R", "R": "L"}
        for (t, s), idx in list(self._sets.items()):
            if t in ("descending_neuron",) or t in ("DNa02", "DNp01", "DNg100", "MDN") or s is None:
                continue
            gain = rng.lognormal(g_in, 0.5, size=(len(idx), 1))               # each type projects its own way
            tgt = self._hid[s]
            W[np.ix_(idx, tgt)] = (rng.random((len(idx), len(tgt))) < 0.15) * gain
        for s in "LR":
            hs = self._hid[s]
            W[np.ix_(hs, self._dn_side[s])] = (rng.random((len(hs), len(self._dn_side[s]))) < 0.10) * rng.lognormal(g_out, 0.5, size=(len(hs), 1))
            W[np.ix_(hs, self._dn_side[other[s]])] = (rng.random((len(hs), len(self._dn_side[other[s]]))) < 0.05) * -0.03
            W[np.ix_(hs, self._hid["L"])] += (rng.random((len(hs), len(self._hid["L"]))) < 0.02) * rng.normal(0, 0.06, size=(len(hs), len(self._hid["L"])))
            W[np.ix_(hs, self._hid["R"])] += (rng.random((len(hs), len(self._hid["R"]))) < 0.02) * rng.normal(0, 0.06, size=(len(hs), len(self._hid["R"])))
        self.W = W
        self.bias = (rng.random(p) * 0.12).astype(np.float32)
        self.v = np.zeros(p, np.float32)
        self.spk = np.zeros(p, np.float32)
        self.rng = rng

    def cells(self, types: Sequence[str], side: Optional[str] = None) -> np.ndarray:
        out = [self._sets[(t, side)] for t in types if (t, side) in self._sets]
        if not out:
            raise KeyError(f"unknown cell type {list(types)} side={side}")
        return np.concatenate(out)

    def step(self, inject=()):
        drive = self.spk @ self.W + self.bias + self.rng.normal(0, 0.12, self.n).astype(np.float32)
        for idx, amount in inject:
            drive[np.asarray(idx)] += amount
        self.v = 0.75 * self.v + drive
        fired = self.v > 1.0
        self.v[fired] = 0.0
        self.spk = fired.astype(np.float32)
        return np.flatnonzero(fired)


# ====================================================================== mock arena
PARRY_WINDOW = (0.03, 0.25)      # a parry press this long before impact counts
JUMP_WINDOW = (0.08, 0.45)
GUARD_MIN = 0.12                 # the guard must have been up this long
FRONT = math.radians(100)


class MockBotw:
    """Lock-step arena.  ``serve`` sends one telemetry packet per frame and waits for the action."""

    def __init__(self, seed: int = 0, fps: int = 30, hp_max: float = 15.0, codec: str = "json"):
        self.rng = np.random.default_rng(seed)
        self.dt = 1.0 / fps
        self.codec = codec
        self.hp_max = hp_max
        self.hp = hp_max
        self.t = 0.0
        self.frame = 0
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.shield_up = False
        self.shield_since = -9.0
        self.air_until = -9.0
        self.press = {"parry": -9.0, "jump": -9.0, "attack": -9.0}
        self.threat: Optional[dict] = None
        self.next_spawn = 0.8
        self.events: List[GameEvent] = []
        self.audio: List[AudioCue] = []
        self.outcomes: collections.Counter = collections.Counter()
        self.n_outcomes = 0
        self._id = 0

    # ------------------------------------------------------------------ world
    def _spawn(self):
        self._id += 1
        if self.rng.random() < 0.6:                                   # melee enemy
            b = math.radians(self.rng.uniform(-150, 150))
            p = self.pos + 8.0 * np.array([math.sin(b), math.cos(b)])
            self.threat = dict(kind="melee", id=self._id, p=p, phase="approach", t_strike=None, hp=2, born=self.t)
        else:                                                        # arrow
            b = math.radians(self.rng.uniform(-60, 60))
            p = self.pos + 22.0 * np.array([math.sin(b), math.cos(b)])
            v = (self.pos - p) / np.linalg.norm(self.pos - p) * 14.0
            self.threat = dict(kind="arrow", id=self._id, p=p, v=v, link0=self.pos.copy(), born=self.t,
                               t_impact=self.t + 22.0 / 14.0, phase="fly")
            self.audio.append(AudioCue(AudioKind.WHOOSH, b, 0.7))

    def _emit(self, kind: EventKind, amount: float = 0.0, bearing: float = float("nan")):
        self.events.append(GameEvent(kind, t=self.t, amount=amount, entity_id=self.threat["id"] if self.threat else -1, bearing=bearing))
        self.outcomes[kind.name] += 1
        self.n_outcomes += 1

    def _resolve(self, frontal: bool, displaced: bool, damage: float, bearing: float):
        t = self.t
        parry_ok = PARRY_WINDOW[0] <= t - self.press["parry"] <= PARRY_WINDOW[1]
        block_ok = self.shield_up and t - self.shield_since >= GUARD_MIN and t >= self.air_until
        jump_ok = JUMP_WINDOW[0] <= t - self.press["jump"] <= JUMP_WINDOW[1]
        if frontal and parry_ok:
            self._emit(EventKind.PARRY, bearing=bearing)
        elif frontal and block_ok:
            self._emit(EventKind.BLOCK, bearing=bearing)
        elif jump_ok:
            self._emit(EventKind.PERFECT_DODGE, bearing=bearing)
        elif displaced:
            self._emit(EventKind.DODGE, bearing=bearing)
        else:
            self.hp -= damage
            self._emit(EventKind.HIT_TAKEN, damage, bearing)
            if self.hp <= 0:
                self.hp = self.hp_max

    def advance(self, act: ActionState):
        dt = self.dt
        self.t += dt
        self.frame += 1
        t = self.t
        # Link
        if act.jump and t >= self.air_until:
            self.air_until = t + 0.4
            self.press["jump"] = t
        if act.parry and t >= self.air_until:
            self.press["parry"] = t
        if act.attack:
            self.press["attack"] = t
        want_guard = (act.shield or act.parry) and t >= self.air_until
        if want_guard and not self.shield_up:
            self.shield_since = t
        self.shield_up = want_guard
        self.vel = np.array([act.move_x, act.move_z]) * (2.0 if self.shield_up else 4.0)
        self.pos = self.pos + self.vel * dt

        th = self.threat
        if th is None:
            if t >= self.next_spawn:
                self._spawn()
            return
        if th["kind"] == "melee":
            rel = th["p"] - self.pos
            d = float(np.linalg.norm(rel))
            if th["phase"] == "approach":
                th["p"] = th["p"] - rel / max(d, 1e-6) * 4.0 * dt
                if d <= 2.6:
                    th["phase"], th["t_strike"] = "windup", t + 0.5
                    self.audio.append(AudioCue(AudioKind.METAL, math.atan2(rel[0], rel[1]), 0.8))
            elif th["phase"] == "windup":
                if act.attack and abs(math.atan2(rel[0], rel[1])) < math.radians(60) and d < 2.8 and t - self.press["attack"] < 1e-9:
                    th["hp"] -= 1
                    self._emit(EventKind.HIT_LANDED, 1.0)
                    if th["hp"] <= 0:
                        self.threat, self.next_spawn = None, t + self.rng.uniform(0.6, 1.4)
                        return
                if t >= th["t_strike"]:
                    bearing = math.atan2(rel[0], rel[1])
                    self._resolve(abs(bearing) <= FRONT, d >= 3.4, 2.0, bearing)
                    th["phase"], th["t_done"] = "recover", t + 0.5
            elif th["phase"] == "recover" and t >= th["t_done"]:
                self.threat, self.next_spawn = None, t + self.rng.uniform(0.6, 1.4)
        else:
            th["p"] = th["p"] + th["v"] * dt
            if t >= th["t_impact"]:
                u = th["v"] / np.linalg.norm(th["v"])
                shift = self.pos - th["link0"]
                lateral = abs(float(shift[0] * -u[1] + shift[1] * u[0]))
                origin = th["link0"] - th["v"] / 14.0 * 22.0
                rel = origin - th["link0"]
                bearing = math.atan2(rel[0], rel[1])
                self._resolve(abs(bearing) <= FRONT, lateral >= 1.0, 1.0, bearing)
                self.threat, self.next_spawn = None, t + self.rng.uniform(0.6, 1.4)

    def telemetry(self) -> Telemetry:
        ents = []
        th = self.threat
        if th is not None:
            rel = th["p"] - self.pos
            if th["kind"] == "melee":
                windup = 0.0
                if th["phase"] == "windup":
                    windup = float(np.clip(1.0 - (th["t_strike"] - self.t) / 0.5, 0.0, 1.0))
                ents.append(Entity(id=th["id"], kind=EntityKind.ENEMY, pos=(float(rel[0]), 0.0, float(rel[1])), vel=None,
                                   radius=0.6, windup=windup, attacking=windup > 0.6))
            else:
                rv = th["v"] - self.vel
                ents.append(Entity(id=th["id"], kind=EntityKind.PROJECTILE, pos=(float(rel[0]), 1.0, float(rel[1])),
                                   vel=(float(rv[0]), 0.0, float(rv[1])), radius=0.1))
        tel = Telemetry(t=self.t, dt=self.dt, frame=self.frame,
                        link=LinkState(hp=self.hp, hp_max=self.hp_max, stamina=1.0, speed=float(np.linalg.norm(self.vel)),
                                       shield_up=self.shield_up, on_ground=self.t >= self.air_until),
                        entities=ents, audio=self.audio, events=self.events)
        self.audio, self.events = [], []
        return tel

    # ------------------------------------------------------------------ lock-step server
    def serve(self, send_frame: Callable[[bytes], None], recv_frame: Callable[[], Optional[bytes]],
              should_stop: Callable[[], bool] = lambda: False) -> None:
        """Send telemetry, wait for the action, advance one frame; until told to stop or the peer goes away."""
        enc = encode_binary if self.codec == "binary" else encode_json
        while not should_stop():
            send_frame(enc(self.telemetry()))
            payload = None
            while payload is None and not should_stop():
                payload = recv_frame()
            if payload is None:
                return
            self.advance(decode_action(payload))
