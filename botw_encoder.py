"""botw_encoder.py -- Breath of the Wild telemetry in, fly sensory-neuron input out.

This module is the "senses" half of the agent.  It never decides anything: it turns what the
game reports into (neuron indices, amount) pairs for ``FlyBrain.step(inject=...)``.

------------------------------------------------------------------------------------------
TELEMETRY CONTRACT (C++ named-pipe bridge  ->  Python)
------------------------------------------------------------------------------------------
One packet per game frame.  Coordinates are EGOCENTRIC and use the same axes as the stick
commands that main.py sends back:   +x = right,  +y = up,  +z = forward,  metres.
(If your hook reports world coordinates, rotate them into Link's frame on the C++ side.)

JSON packet (UTF-8; extra keys are ignored, missing keys take the defaults shown):

    {"v": 1, "t": 12.345, "dt": 0.0333, "frame": 411,
     "link":     {"hp": 12, "hp_max": 15, "stamina": 1.0, "speed": 2.1,
                  "shield_up": false, "has_shield": true, "on_ground": true,
                  "targeting": false, "stunned": false},
     "entities": [{"id": 7, "kind": "enemy",             # enemy | projectile | laser | other
                   "pos": [-2.0, 0.0, 4.0],              # ego frame, metres
                   "vel": [0.5, 0.0, -3.0],              # velocity RELATIVE to Link, optional
                   "radius": 0.6, "windup": 0.4,         # windup 0..1 = progress of the attack telegraph
                   "attacking": false, "tti": null,      # tti = seconds to impact if the game knows it
                   "locked": false}],                    # Z-target lock
     "audio":    [{"kind": "metal", "az": -0.8, "level": 0.7}],     # az: rad, 0 ahead, + right
     "events":   [{"type": "hit_taken", "t": 12.30, "amount": 1.0, "id": 7, "bearing": -0.5}]}

``events`` are the OUTCOMES the hindsight labeller learns from.  "type" is one of
hit_taken, parry, block, perfect_dodge, dodge, hit_landed, kill, threat_expired (or the integer
values of EventKind).  If the bridge sends no events at all, a drop in ``link.hp`` is turned
into a hit_taken event (OutcomeInference) so the agent can still learn from damage alone.

BINARY packet (little-endian, starts with b"BFLY"; decode_packet() tells the two apart):

    header   "<4sBBBBIff"  magic, version, n_entities, n_audio, n_events, frame, t, dt
    link     "<4fI"        hp, hp_max, stamina, speed, flags
                           flags: b0 shield_up, b1 has_shield, b2 on_ground, b3 targeting, b4 stunned
    entity   "<HBB9f"      id, kind, flags (b0 attacking, b1 locked, b2 has_vel),
                           x y z, vx vy vz, radius, windup, tti (NaN = unknown)
    audio    "<Bff"        kind, azimuth, level
    event    "<BHfff"      kind, entity_id (0xFFFF = none), t, amount, bearing (NaN = unknown)

``encode_binary`` / ``encode_json`` below are the reference writers (the simulator uses them).

------------------------------------------------------------------------------------------
WHAT DRIVES WHICH NEURONS   (a choice made here -- the fly.ai README leaves encoders to the task)
------------------------------------------------------------------------------------------
Same four visual cell types and roles as the fly.ai SSH Fighter bot, one population per eye:

    LPLC2  looming          size and approach speed of anything near (enemy, arrow, beam)
    LC4    fast looming     imminent hits: attack windup, short time-to-impact
    LPLC1  small objects    projectiles (arrows, rocks, laser bolts)
    LC10a  chase target     where the enemies are, stronger when close (drives DNa02 steering)

Each object is split between the eyes by azimuth (see ``_eye_weights``).  Audio cues drive the
Johnston's-organ (hearing) neurons on the side the sound came from; if no such neurons can be
found, they drive LC4 on that side instead.  Link's HP scales the looming/threat gain (a hurt
fly startles harder).  Everything else about Link's state (airborne, stunned, shield) is a
physical constraint on the OUTPUT side and lives in botw_readout.ActionDecoder.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field, fields
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

NAN = float("nan")
SCHEMA_VERSION = 1
BIN_MAGIC = b"BFLY"


# ====================================================================== enums and records
class EntityKind(IntEnum):
    OTHER = 0
    ENEMY = 1
    PROJECTILE = 2
    LASER = 3


class AudioKind(IntEnum):
    GENERIC = 0
    METAL = 1       # weapon swing / clang
    VOICE = 2       # growl, shout
    CHARGE = 3      # Guardian laser charging whine
    FOOTSTEP = 4
    WHOOSH = 5      # arrow or rock release


class EventKind(IntEnum):
    HIT_TAKEN = 1
    PARRY = 2
    BLOCK = 3
    PERFECT_DODGE = 4
    DODGE = 5            # a threat passed without damage after Link moved out of its way
    HIT_LANDED = 6
    KILL = 7
    THREAT_EXPIRED = 8   # a threat ended without touching Link and without Link doing anything


@dataclass
class LinkState:
    hp: float = 1.0
    hp_max: float = 1.0
    stamina: float = 1.0
    speed: float = 0.0
    shield_up: bool = False
    has_shield: bool = True
    on_ground: bool = True
    targeting: bool = False
    stunned: bool = False


@dataclass
class Entity:
    id: int = 0
    kind: int = EntityKind.ENEMY
    pos: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    vel: Optional[Tuple[float, float, float]] = None
    radius: float = 0.5
    windup: float = 0.0
    attacking: bool = False
    tti: float = NAN
    locked: bool = False


@dataclass
class AudioCue:
    kind: int = AudioKind.GENERIC
    azimuth: float = 0.0
    level: float = 0.0


@dataclass
class GameEvent:
    kind: int
    t: float = 0.0
    amount: float = 0.0
    entity_id: int = -1
    bearing: float = NAN


@dataclass
class Telemetry:
    t: float = 0.0
    dt: float = 1.0 / 30.0
    frame: int = 0
    link: LinkState = field(default_factory=LinkState)
    entities: List[Entity] = field(default_factory=list)
    audio: List[AudioCue] = field(default_factory=list)
    events: List[GameEvent] = field(default_factory=list)


# ====================================================================== codecs
def _enum(cls, v, default=None):
    if isinstance(v, str):
        name = v.strip().upper()
        return cls[name] if name in cls.__members__ else default
    try:
        return cls(int(v))
    except (ValueError, TypeError):
        return default


def _vec3(v) -> Optional[Tuple[float, float, float]]:
    if v is None:
        return None
    try:
        x, y, z = (float(c) for c in v)
    except (TypeError, ValueError):
        return None
    return (x, y, z)


def _f(v, default):
    return default if v is None else float(v)


def telemetry_from_dict(d: dict) -> Telemetry:
    lk = d.get("link") or {}
    hp = _f(lk.get("hp"), 1.0)
    link = LinkState(
        hp=hp, hp_max=_f(lk.get("hp_max"), max(hp, 1.0)), stamina=_f(lk.get("stamina"), 1.0),
        speed=_f(lk.get("speed"), 0.0), shield_up=bool(lk.get("shield_up", False)),
        has_shield=bool(lk.get("has_shield", True)), on_ground=bool(lk.get("on_ground", True)),
        targeting=bool(lk.get("targeting", False)), stunned=bool(lk.get("stunned", False)))
    ents = []
    for e in d.get("entities") or []:
        pos = _vec3(e.get("pos"))
        if pos is None:
            continue
        ents.append(Entity(
            id=int(e.get("id", len(ents))), kind=_enum(EntityKind, e.get("kind", 1), EntityKind.OTHER),
            pos=pos, vel=_vec3(e.get("vel")), radius=_f(e.get("radius"), 0.5),
            windup=_f(e.get("windup"), 0.0), attacking=bool(e.get("attacking", False)),
            tti=_f(e.get("tti"), NAN), locked=bool(e.get("locked", False))))
    audio = [AudioCue(kind=_enum(AudioKind, a.get("kind", 0), AudioKind.GENERIC),
                      azimuth=_f(a.get("az", a.get("azimuth")), 0.0), level=_f(a.get("level"), 0.0))
             for a in d.get("audio") or []]
    events = []
    for ev in d.get("events") or []:
        kind = _enum(EventKind, ev.get("type", ev.get("kind")), None)
        if kind is None:
            continue
        events.append(GameEvent(kind=kind, t=_f(ev.get("t"), _f(d.get("t"), 0.0)),
                                amount=_f(ev.get("amount"), 0.0), entity_id=int(ev.get("id", -1)),
                                bearing=_f(ev.get("bearing"), NAN)))
    return Telemetry(t=_f(d.get("t"), 0.0), dt=_f(d.get("dt"), 1.0 / 30.0), frame=int(d.get("frame", 0)),
                     link=link, entities=ents, audio=audio, events=events)


def telemetry_to_dict(t: Telemetry) -> dict:
    def clean(x):
        return None if isinstance(x, float) and math.isnan(x) else x
    return {
        "v": SCHEMA_VERSION, "t": t.t, "dt": t.dt, "frame": t.frame,
        "link": {"hp": t.link.hp, "hp_max": t.link.hp_max, "stamina": t.link.stamina,
                 "speed": t.link.speed, "shield_up": t.link.shield_up, "has_shield": t.link.has_shield,
                 "on_ground": t.link.on_ground, "targeting": t.link.targeting, "stunned": t.link.stunned},
        "entities": [{"id": e.id, "kind": int(e.kind), "pos": list(e.pos),
                      "vel": None if e.vel is None else list(e.vel), "radius": e.radius,
                      "windup": e.windup, "attacking": e.attacking, "tti": clean(e.tti),
                      "locked": e.locked} for e in t.entities],
        "audio": [{"kind": int(a.kind), "az": a.azimuth, "level": a.level} for a in t.audio],
        "events": [{"type": int(v.kind), "t": v.t, "amount": v.amount, "id": v.entity_id,
                    "bearing": clean(v.bearing)} for v in t.events],
    }


def encode_json(t: Telemetry) -> bytes:
    return json.dumps(telemetry_to_dict(t), separators=(",", ":")).encode("utf-8")


_HDR = struct.Struct("<4sBBBBIff")
_LNK = struct.Struct("<4fI")
_ENT = struct.Struct("<HBB9f")
_AUD = struct.Struct("<Bff")
_EVT = struct.Struct("<BHfff")


def encode_binary(t: Telemetry) -> bytes:
    out = [_HDR.pack(BIN_MAGIC, SCHEMA_VERSION, len(t.entities), len(t.audio), len(t.events),
                     t.frame & 0xFFFFFFFF, t.t, t.dt)]
    k = t.link
    flags = (k.shield_up << 0) | (k.has_shield << 1) | (k.on_ground << 2) | (k.targeting << 3) | (k.stunned << 4)
    out.append(_LNK.pack(k.hp, k.hp_max, k.stamina, k.speed, flags))
    for e in t.entities:
        v = e.vel if e.vel is not None else (0.0, 0.0, 0.0)
        out.append(_ENT.pack(e.id & 0xFFFF, int(e.kind),
                             (e.attacking << 0) | (e.locked << 1) | ((e.vel is not None) << 2),
                             *e.pos, *v, e.radius, e.windup, e.tti))
    for a in t.audio:
        out.append(_AUD.pack(int(a.kind), a.azimuth, a.level))
    for v in t.events:
        out.append(_EVT.pack(int(v.kind), 0xFFFF if v.entity_id < 0 else v.entity_id & 0xFFFF,
                             v.t, v.amount, v.bearing))
    return b"".join(out)


def _decode_binary(buf: bytes) -> Telemetry:
    magic, ver, ne, na, nv, frame, t, dt = _HDR.unpack_from(buf, 0)
    if magic != BIN_MAGIC or ver != SCHEMA_VERSION:
        raise ValueError(f"unsupported binary telemetry (magic={magic!r}, version={ver})")
    off = _HDR.size
    hp, hp_max, stamina, speed, fl = _LNK.unpack_from(buf, off)
    off += _LNK.size
    link = LinkState(hp=hp, hp_max=hp_max, stamina=stamina, speed=speed, shield_up=bool(fl & 1),
                     has_shield=bool(fl & 2), on_ground=bool(fl & 4), targeting=bool(fl & 8),
                     stunned=bool(fl & 16))
    ents = []
    for _ in range(ne):
        i, kind, ef, x, y, z, vx, vy, vz, radius, windup, tti = _ENT.unpack_from(buf, off)
        off += _ENT.size
        ents.append(Entity(id=i, kind=_enum(EntityKind, kind, EntityKind.OTHER), pos=(x, y, z),
                           vel=(vx, vy, vz) if ef & 4 else None, radius=radius, windup=windup,
                           attacking=bool(ef & 1), tti=tti, locked=bool(ef & 2)))
    audio = []
    for _ in range(na):
        k, az, lv = _AUD.unpack_from(buf, off)
        off += _AUD.size
        audio.append(AudioCue(kind=_enum(AudioKind, k, AudioKind.GENERIC), azimuth=az, level=lv))
    events = []
    for _ in range(nv):
        k, eid, et, amt, brg = _EVT.unpack_from(buf, off)
        off += _EVT.size
        kind = _enum(EventKind, k, None)
        if kind is not None:
            events.append(GameEvent(kind=kind, t=et, amount=amt, entity_id=-1 if eid == 0xFFFF else eid, bearing=brg))
    return Telemetry(t=t, dt=dt, frame=frame, link=link, entities=ents, audio=audio, events=events)


def decode_packet(payload: bytes) -> Telemetry:
    """JSON or binary, told apart by the first bytes.  Raises ValueError on garbage."""
    if payload[:4] == BIN_MAGIC:
        return _decode_binary(payload)
    try:
        return telemetry_from_dict(json.loads(payload))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"telemetry packet is neither JSON nor binary: {exc}") from exc


def merge_packets(packets: Sequence[Telemetry]) -> Telemetry:
    """Collapse a backlog into one packet: newest state, but every event and cue is kept and
    the elapsed time is summed, so outcomes are never lost when the brain falls behind."""
    last = packets[-1]
    if len(packets) == 1:
        return last
    return Telemetry(t=last.t, dt=sum(p.dt for p in packets), frame=last.frame, link=last.link,
                     entities=last.entities, audio=[a for p in packets for a in p.audio],
                     events=[e for p in packets for e in p.events])


class OutcomeInference:
    """If the bridge sends no hit_taken event, turn an HP drop into one.  A drop that follows an explicit
    hit_taken event within ``grace`` seconds is the same hit and is not counted twice."""

    def __init__(self, grace: float = 0.3):
        self.prev_hp: Optional[float] = None
        self.last_hit_t = -1e9
        self.grace = grace

    def __call__(self, tel: Telemetry) -> List[GameEvent]:
        events = list(tel.events)
        if any(e.kind == EventKind.HIT_TAKEN for e in events):
            self.last_hit_t = tel.t
        hp = tel.link.hp
        if (self.prev_hp is not None and hp < self.prev_hp - 1e-3 and tel.t - self.last_hit_t > self.grace):
            events.append(GameEvent(EventKind.HIT_TAKEN, t=tel.t, amount=self.prev_hp - hp))
            self.last_hit_t = tel.t
        self.prev_hp = hp
        return events


# ====================================================================== geometry helpers
def evasion_vector(pos: Sequence[float], vel: Optional[Sequence[float]] = None) -> np.ndarray:
    """Unit (x, z) step that gets Link out of a threat's way.  Pure geometry, used only to
    write hindsight labels: for something flying along a line, step perpendicular to the line;
    for a melee attacker, back off and sidestep away from the side it stands on."""
    x, z = float(pos[0]), float(pos[2])
    d = math.hypot(x, z)
    if d < 1e-6:
        return np.array([1.0, 0.0], np.float32)
    if vel is not None:
        vx, vz = float(vel[0]), float(vel[2])
        sp = math.hypot(vx, vz)
        if sp > 1.0:
            ux, uz = vx / sp, vz / sp
            t = -(x * ux + z * uz)
            cx, cz = x + t * ux, z + t * uz           # closest point of the flight line to Link
            c = math.hypot(cx, cz)
            if c > 0.3:
                return np.array([-cx / c, -cz / c], np.float32)
            px, pz = -uz, ux                           # dead centre: pick a side
            return np.array([px, pz], np.float32)
    lat = -1.0 if x > 0.2 else 1.0
    v = np.array([-x / d * 0.6 + lat * 0.8, -z / d * 0.6], np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-6)


# ====================================================================== neurons
def as_index_array(x) -> np.ndarray:
    """Whatever ``brain.cells`` / ``brain.step`` hand back (numpy, CuPy, torch, list, bool mask)
    as a sorted int64 numpy index array."""
    if x is None:
        return np.empty(0, np.int64)
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        x = x.cpu().numpy()
    elif hasattr(x, "get") and hasattr(x, "device"):
        x = x.get()
    a = np.asarray(x)
    if a.dtype == bool:
        a = np.flatnonzero(a)
    return np.unique(a.astype(np.int64, copy=False).ravel())


VISUAL_TYPES = {"loom": "LPLC2", "threat": "LC4", "proj": "LPLC1", "chase": "LC10a"}
# Johnston's organ (the fly's ear).  Naming differs between connectome releases, so try several.
AUDIO_TYPE_CANDIDATES = ("JO-A", "JO-B", "JO-C", "JO-E", "JO-F", "JO", "JON")
# Identified command neurons, used by `check` to show that the court reaches the output side.
NAMED_DN = ("DNa02", "DNp01", "DNg100", "MDN")


def _cells(brain, types: Sequence[str], side: Optional[str], errors: Optional[list] = None) -> np.ndarray:
    try:
        idx = brain.cells(list(types), side=side) if side else brain.cells(list(types))
    except Exception as exc:                      # unknown type name, wrong build of the connectome...
        if errors is not None:
            errors.append(f"cells({list(types)}, side={side}): {type(exc).__name__}: {exc}")
        return np.empty(0, np.int64)
    return as_index_array(idx)


class CellMap:
    """The neuron index sets the encoder writes to and the trace reads from."""

    def __init__(self, brain, audio_types: Optional[Sequence[str]] = None):
        self.errors: List[str] = []
        self.eye = {(ch, s): _cells(brain, [typ], s, self.errors)
                    for ch, typ in VISUAL_TYPES.items() for s in "LR"}
        cands = tuple(audio_types) if audio_types else AUDIO_TYPE_CANDIDATES
        self.audio_found: List[str] = []
        self.audio: Dict[str, np.ndarray] = {}
        for s in "LR":
            parts = []
            for name in cands:
                got = _cells(brain, [name], s)
                if got.size:
                    parts.append(got)
                    if name not in self.audio_found:
                        self.audio_found.append(name)
            self.audio[s] = np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)
        self.dn = _cells(brain, ["descending_neuron"], None, self.errors)
        self.dn_side = {s: _cells(brain, ["descending_neuron"], s) for s in "LR"}
        self.named = {n: {s: _cells(brain, [n], s) for s in "LR"} for n in NAMED_DN}
        for n in NAMED_DN:                        # unsided neurons (MDN may have no side) fall back to "any"
            if not any(self.named[n][s].size for s in "LR"):
                self.named[n]["LR"] = _cells(brain, [n], None)

    def describe(self) -> str:
        rows = []
        for ch, typ in VISUAL_TYPES.items():
            rows.append(f"  {typ:<6} ({ch:<6}) L={self.eye[(ch, 'L')].size:<4} R={self.eye[(ch, 'R')].size:<4}")
        a = f"{'+'.join(self.audio_found) or 'none -> audio falls back to LC4'}"
        rows.append(f"  hearing  L={self.audio['L'].size:<4} R={self.audio['R'].size:<4} ({a})")
        rows.append(f"  descending neurons: {self.dn.size}")
        return "\n".join(rows)

    def problems(self) -> List[str]:
        out = [f"no neurons for {VISUAL_TYPES[ch]} on side {s}" for (ch, s), idx in self.eye.items() if idx.size == 0]
        if self.dn.size == 0:
            out.append("no descending neurons found")
        return out + self.errors


# ====================================================================== encoder
@dataclass
class EncoderConfig:
    # injected amount per neuron when a channel is fully driven (the README's tests used 0.8)
    amp_loom: float = 0.8
    amp_threat: float = 0.8
    amp_proj: float = 0.6
    amp_chase: float = 0.7
    amp_audio: float = 0.5
    loom_size: float = 0.6        # share of the looming drive that comes from object size vs approach speed
                                  # (fly.ai SSH Fighter sweep: 0.6 is the smallest value on the plateau)
    n_bins: int = 1               # azimuth sub-populations per eye; 1 = plain left/right
    eye_axis_deg: float = 60.0    # each eye looks this far to its own side
    eye_kappa: float = 1.2        # width of each eye's field of view (von Mises)
    audio_axis_deg: float = 90.0
    range_enemy: float = 14.0     # metres at which an enemy starts to count
    range_proj: float = 30.0
    v_ref_enemy: float = 6.0      # m/s of approach mapped to 1.0
    v_ref_proj: float = 25.0
    size_ref: float = 0.5         # angular size (rad) mapped to 1.0
    ttc_ref: float = 1.2          # time-to-impact (s) below which urgency starts to rise
    hp_gain: float = 0.5          # extra gain on loom/threat at zero HP
    threat_floor: float = 0.3     # windup counts at least this much even when far
    saturation: float = 1.5       # tanh steepness: soft cap so strong input cannot swamp the neurons
    max_entities: int = 32        # only the nearest this many objects are encoded


def _coerce(cur, text: str):
    v = text.strip()
    if isinstance(cur, bool):
        return v.lower() in ("1", "true", "yes", "on")
    if isinstance(cur, tuple):                       # ranks=5:20:60
        elem = type(cur[0]) if cur else float
        return tuple(elem(float(x)) for x in v.split(":") if x)
    return type(cur)(float(v))


def parse_overrides(cfg, text: Optional[str]):
    """'amp_loom=0.9,loom_size=0.4' -> cfg with those fields changed (same idea as fly.ai --encoder).
    Tuple fields take colon-separated values: 'ranks=5:20:60'."""
    if not text:
        return cfg
    names = {f.name for f in fields(cfg)}
    for item in text.split(","):
        if not item.strip():
            continue
        key, _, val = item.partition("=")
        key = key.strip()
        if key not in names:
            raise ValueError(f"unknown parameter {key!r}; choose from {sorted(names)}")
        setattr(cfg, key, _coerce(getattr(cfg, key), val))
    return cfg


@dataclass
class Percept:
    inject: List[Tuple[np.ndarray, float]]
    summary: Dict[str, float]
    # The fields below describe the world for the hindsight labeller, exploration timing and the HUD.
    # The action decoder never reads them: only neural activity drives the actions.
    threat: float = 0.0
    bearing: float = NAN
    evade: Optional[np.ndarray] = None
    enemy_dist: float = NAN
    enemy_az: float = NAN


def _split(idx: np.ndarray, n_bins: int, rng) -> List[np.ndarray]:
    if idx.size == 0:
        return []
    nb = max(1, min(n_bins, idx.size))
    return [np.sort(p) for p in np.array_split(rng.permutation(idx), nb)]


class BotwEncoder:
    def __init__(self, cells: CellMap, cfg: Optional[EncoderConfig] = None, seed: int = 0):
        self.cfg = cfg or EncoderConfig()
        self.cells = cells
        rng = np.random.default_rng(seed)
        # NOTE: the split of a cell type into azimuth bins is arbitrary with respect to real retinotopy
        # (the documented flybrain calls do not expose it).  It is a fixed random partition, which
        # makes the input richer for the readout; keep n_bins=1 for the plain left/right code.
        self.bins = {k: _split(idx, self.cfg.n_bins, rng) for k, idx in cells.eye.items()}
        self._prev_d: Dict[int, float] = {}
        self._prev_closing: Dict[int, float] = {}

    # ------------------------------------------------------------------ helpers
    def _eye_weights(self, az: np.ndarray, axis: float):
        k = self.cfg.eye_kappa
        return np.exp(k * (np.cos(az + axis) - 1.0)), np.exp(k * (np.cos(az - axis) - 1.0))

    @staticmethod
    def _bin_weights(az: np.ndarray, nb: int) -> np.ndarray:
        if nb == 1:
            return np.ones((1, az.size))
        centres = -np.pi + (np.arange(nb) + 0.5) * 2 * np.pi / nb
        w = np.exp(1.6 * nb / 2.0 * (np.cos(az[None, :] - centres[:, None]) - 1.0))
        return w / np.maximum(w.sum(0, keepdims=True), 1e-9)

    def _sat(self, x):
        return np.tanh(self.cfg.saturation * x)

    def _emit(self, inject, summary, name, side, amp, drive_per_entity, w_eye, bw):
        pops = self.bins.get((name, side), [])
        if not pops:
            summary[f"{name}_{side}"] = 0.0
            return
        nb = len(pops)
        bwp = bw if bw.shape[0] == nb else self._bin_weights(self._az, nb)
        level = bwp @ (drive_per_entity * w_eye)                  # (bins,)
        # With nb bins each bin holds 1/nb of the neurons and gets only its share of the object's drive, so
        # scale by nb: the total input then stays within ~30% of the plain left/right code.
        amounts = amp * self._sat(nb * level)
        summary[f"{name}_{side}"] = float(amounts.max())
        for idx, a in zip(pops, amounts):
            if a > 1e-3:
                inject.append((idx, float(a)))

    # ------------------------------------------------------------------ main entry
    def encode(self, tel: Telemetry) -> Percept:
        c = self.cfg
        inject: List[Tuple[np.ndarray, float]] = []
        summary: Dict[str, float] = {}
        gain = 1.0 + c.hp_gain * (1.0 - min(1.0, max(0.0, tel.link.hp / max(tel.link.hp_max, 1e-6))))
        th = math.radians(c.eye_axis_deg)
        threat, bearing, evade = 0.0, NAN, None
        enemy_d, enemy_az = NAN, NAN
        ents = tel.entities
        if len(ents) > c.max_entities:
            ents = sorted(ents, key=lambda e: e.pos[0] ** 2 + e.pos[1] ** 2 + e.pos[2] ** 2)[: c.max_entities]
        n = len(ents)
        self._az = np.zeros(n)

        if n:
            P = np.array([e.pos for e in ents], float)
            kind = np.array([int(e.kind) for e in ents])
            d = np.maximum(np.linalg.norm(P, axis=1), 0.1)
            az = np.arctan2(P[:, 0], P[:, 2])
            self._az = az
            unit = P / d[:, None]
            vel = np.array([e.vel if e.vel is not None else (NAN, NAN, NAN) for e in ents], float)
            has_v = np.isfinite(vel).all(1)
            closing = np.where(has_v, -(vel * unit).sum(1), 0.0)
            ids = [e.id for e in ents]
            for i in np.flatnonzero(~has_v):                       # no velocity from the game: differentiate
                pd = self._prev_d.get(ids[i])
                if pd is not None and tel.dt > 1e-4:
                    raw = (pd - d[i]) / tel.dt
                    closing[i] = 0.5 * raw + 0.5 * self._prev_closing.get(ids[i], raw)
            self._prev_d = {ids[i]: float(d[i]) for i in range(n)}
            self._prev_closing = {ids[i]: float(closing[i]) for i in range(n)}

            is_proj = (kind == EntityKind.PROJECTILE) | (kind == EntityKind.LASER)
            is_enemy = kind == EntityKind.ENEMY
            rng_ = np.where(is_proj, c.range_proj, c.range_enemy)
            vref = np.where(is_proj, c.v_ref_proj, c.v_ref_enemy)
            prox = np.clip(1.0 - d / rng_, 0.0, 1.0)
            clos = np.clip(closing / vref, 0.0, 1.0)
            size = np.clip(2.0 * np.arctan2([e.radius for e in ents], np.maximum(d, 0.2)) / c.size_ref, 0.0, 1.0)
            tti_given = np.array([e.tti for e in ents], float)
            tti = np.where(np.isfinite(tti_given) & (tti_given >= 0), tti_given,
                           np.where(closing > 0.05, d / np.maximum(closing, 0.05), np.inf))
            urg = np.where(np.isfinite(tti), np.clip(1.0 - tti / c.ttc_ref, 0.0, 1.0), 0.0)
            wind = np.array([max(e.windup, 0.6) if e.attacking else e.windup for e in ents], float).clip(0, 1)
            locked = np.array([e.locked for e in ents], float)

            loom = prox * ((1.0 - c.loom_size) * clos + c.loom_size * size)
            thr = np.maximum(wind * np.maximum(prox, c.threat_floor), urg)
            proj = np.where(is_proj, prox * clos + 0.25 * prox, 0.0)
            chase = np.where(is_enemy, (0.05 + 0.95 * prox) * (1.0 + 0.5 * locked), 0.0)

            wl, wr = self._eye_weights(az, th)
            bw = self._bin_weights(az, c.n_bins)
            for side, w_eye in (("L", wl), ("R", wr)):
                self._emit(inject, summary, "loom", side, c.amp_loom * gain, loom, w_eye, bw)
                self._emit(inject, summary, "threat", side, c.amp_threat * gain, thr, w_eye, bw)
                self._emit(inject, summary, "proj", side, c.amp_proj, proj, w_eye, bw)
                self._emit(inject, summary, "chase", side, c.amp_chase, chase, w_eye, bw)

            imminence = np.maximum.reduce([urg, wind * np.maximum(prox, c.threat_floor), prox * clos])
            j = int(np.argmax(imminence))
            if imminence[j] >= 0.05:
                threat, bearing = float(imminence[j]), float(az[j])
                evade = evasion_vector(P[j], ents[j].vel)
            if is_enemy.any():
                k = int(np.argmin(np.where(is_enemy, d, np.inf)))
                enemy_d, enemy_az = float(d[k]), float(az[k])
        else:
            self._prev_d.clear()
            self._prev_closing.clear()

        # hearing: each cue drives the ear on its own side (or LC4 if this brain has no ear neurons)
        if tel.audio:
            az_a = np.array([a.azimuth for a in tel.audio], float)
            lvl = np.array([a.level * AUDIO_GAIN.get(int(a.kind), 0.8) for a in tel.audio], float)
            wl, wr = self._eye_weights(az_a, math.radians(c.audio_axis_deg))
            for side, w in (("L", wl), ("R", wr)):
                amount = c.amp_audio * float(self._sat((lvl * w).sum()))
                idx = self.cells.audio[side]
                if idx.size == 0:
                    idx = self.cells.eye[("threat", side)]
                summary[f"audio_{side}"] = amount
                if amount > 1e-3 and idx.size:
                    inject.append((idx, amount))
        return Percept(inject=inject, summary=summary, threat=threat, bearing=bearing, evade=evade,
                       enemy_dist=enemy_d, enemy_az=enemy_az)


AUDIO_GAIN = {int(AudioKind.GENERIC): 0.8, int(AudioKind.METAL): 1.0, int(AudioKind.VOICE): 0.7,
              int(AudioKind.CHARGE): 1.0, int(AudioKind.FOOTSTEP): 0.4, int(AudioKind.WHOOSH): 0.9}
