"""botw_readout.py -- fly neural activity in, BotW controller actions out, learned in hindsight.

Pipeline (the brain itself never trains; only the linear readout in step 4 is fitted):

    1. SpikeTrace        decaying spike count of the ~1,300 descending neurons, one vector per tick
    2. HindsightLabeller every tick is remembered; when the game reports an OUTCOME the last second
                         of ticks is labelled with what the outcome says about the actions
    3. ExperienceMemory  labelled ticks (features, targets, per-head weights, incident id)
    4. LinearReadout     PCA + weighted ridge per action head, rank and penalty by leave-incident-out
                         cross-validation; binary heads get a threshold tuned on the held-out scores
    5. ActionDecoder     scores -> controller state (thresholds, hysteresis, cooldowns, body limits)

Heads: move_x, move_z (continuous stick), shield (hold), parry, jump, attack (one-shot presses).

HOW OUTCOMES BECOME LABELS  (the BotW counterpart of "where the ball really arrived")
-----------------------------------------------------------------------------------
  parry / block / perfect_dodge / dodge / hit_landed   SUCCESS.  The ticks before it are labelled with
        the actions that were really executed (imitate what worked).  Only presses that fell inside
        the effective timing window are credited; earlier, stray presses get no credit.
  hit_taken   FAILURE.  The impact time is now known, so the ticks before it are relabelled with what
        would have worked: raise the shield ~0.3 s before, press parry ~0.12 s before, jump ~0.28 s
        before, step away from the attacker.  Shield and parry only if the hit came from the front.
        Counterfactual labels weigh less than real successes (``w_cf``); bigger damage weighs more.
  threat_expired   neutral: a looming thing that needed no answer, so every press head is labelled 0.
  quiet ticks      (no threat in sight, no outcome pending) are labelled "do nothing", sampled sparsely.
The lead times are body/game constants (how long it takes the guard to come up, how wide the parry
window is).  They are assumptions: tune them in LabelConfig or by editing the table below.

Only neural activity drives the actions.  The world knowledge used here (impact times, bearings, the
evasion geometry) is used to WRITE LABELS and to time exploration, never to choose an action.
"""
from __future__ import annotations

import collections
import json
import math
import struct
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from botw_encoder import EventKind, GameEvent, LinkState, Percept, as_index_array

NAN = float("nan")

HEADS = ("move_x", "move_z", "shield", "parry", "jump", "attack")
N_HEADS = len(HEADS)
MOVE_X, MOVE_Z, SHIELD, PARRY, JUMP, ATTACK = range(N_HEADS)
MOVE_HEADS = (MOVE_X, MOVE_Z)
PULSE_HEADS = (PARRY, JUMP, ATTACK)
IS_BINARY = np.array([False, False, True, True, True, True])


# ====================================================================== action wire format
ACT_MAGIC = b"BFLA"
_ACT = struct.Struct("<4sBBIff")


@dataclass
class ActionState:
    """What Python tells the game.  Axes are Link's egocentric frame (same as the telemetry).

    shield  level: hold guard while True.
    parry / jump / attack  are ONE-SHOT: True in exactly one packet = one button press.  The C++
    side decides how a press maps to buttons (e.g. parry = guard + A, jump = dodge when locked on).
    """
    move_x: float = 0.0       # -1 left ... +1 right
    move_z: float = 0.0       # -1 back ... +1 forward
    shield: bool = False
    parry: bool = False
    jump: bool = False
    attack: bool = False
    seq: int = 0

    def vector(self) -> np.ndarray:
        return np.array([self.move_x, self.move_z, self.shield, self.parry, self.jump, self.attack], np.float32)

    def to_json(self) -> bytes:
        return json.dumps({"v": 1, "seq": self.seq, "move": [round(self.move_x, 4), round(self.move_z, 4)],
                           "shield": self.shield, "parry": self.parry, "jump": self.jump,
                           "attack": self.attack}, separators=(",", ":")).encode("utf-8")

    def to_binary(self) -> bytes:
        flags = self.shield | (self.parry << 1) | (self.jump << 2) | (self.attack << 3)
        return _ACT.pack(ACT_MAGIC, 1, flags, self.seq & 0xFFFFFFFF, self.move_x, self.move_z)

    @staticmethod
    def from_json(payload: bytes) -> "ActionState":
        d = json.loads(payload)
        mv = d.get("move", [0.0, 0.0])
        return ActionState(float(mv[0]), float(mv[1]), bool(d.get("shield")), bool(d.get("parry")),
                           bool(d.get("jump")), bool(d.get("attack")), int(d.get("seq", 0)))

    @staticmethod
    def from_binary(payload: bytes) -> "ActionState":
        magic, ver, fl, seq, mx, mz = _ACT.unpack_from(payload, 0)
        if magic != ACT_MAGIC:
            raise ValueError("not a BFLA action packet")
        return ActionState(mx, mz, bool(fl & 1), bool(fl & 2), bool(fl & 4), bool(fl & 8), seq)


def decode_action(payload: bytes) -> ActionState:
    return ActionState.from_binary(payload) if payload[:4] == ACT_MAGIC else ActionState.from_json(payload)


# ====================================================================== 1. spike trace
class SpikeTrace:
    """Exponentially decaying spike count per descending neuron (tau = 100 ms, as in fly.ai SSH Fighter)."""

    def __init__(self, dn_idx, dt: float, tau: float = 0.1):
        self.idx = as_index_array(dn_idx)
        self.n = int(self.idx.size)
        self.decay = math.exp(-dt / tau)
        self.v = np.zeros(self.n, np.float32)

    def update(self, fired) -> None:
        self.v *= self.decay
        f = as_index_array(fired)
        if f.size and self.n:
            pos = np.searchsorted(self.idx, f)
            ok = pos < self.n
            pos, f = pos[ok], f[ok]
            self.v[pos[self.idx[pos] == f]] += 1.0

    def vector(self) -> np.ndarray:
        return self.v.copy()

    def reset(self) -> None:
        self.v[:] = 0.0


# ====================================================================== 3. experience memory
class ExperienceMemory:
    """Labelled ticks: features X (float16), targets Y, per-head weights W (0 = no label), incident id G."""

    def __init__(self, n_features: int, max_rows: int = 30000):
        self.d, self.max_rows = int(n_features), int(max_rows)
        self._alloc(min(self.max_rows, 2048))
        self.n = 0
        self.ptr = 0
        self.groups = 0
        self.lock = threading.Lock()

    def _alloc(self, cap: int, old: Optional[Tuple[np.ndarray, ...]] = None, n: int = 0) -> None:
        X = np.zeros((cap, self.d), np.float16)
        Y = np.zeros((cap, N_HEADS), np.float32)
        W = np.zeros((cap, N_HEADS), np.float32)
        G = np.zeros(cap, np.int32)
        if old is not None:
            for new, o in zip((X, Y, W, G), old):
                new[:n] = o[:n]
        self.X, self.Y, self.W, self.G = X, Y, W, G

    def adopt(self, other: "ExperienceMemory") -> None:
        """Take over the contents of another memory (used when loading a saved one into the live object)."""
        with self.lock, other.lock:
            self.d, self.max_rows = other.d, other.max_rows
            self.X, self.Y, self.W, self.G = other.X, other.Y, other.W, other.G
            self.n, self.ptr, self.groups = other.n, other.ptr, other.groups

    def new_group(self) -> int:
        with self.lock:
            self.groups += 1
            return self.groups

    def add(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, g: int) -> None:
        with self.lock:
            if self.n < self.max_rows:
                if self.n == self.X.shape[0]:
                    self._alloc(min(self.max_rows, self.X.shape[0] * 2), (self.X, self.Y, self.W, self.G), self.n)
                i = self.n
                self.n += 1
            else:                                     # full: overwrite the oldest
                i = self.ptr
                self.ptr = (self.ptr + 1) % self.max_rows
            self.X[i], self.Y[i], self.W[i], self.G[i] = x, y, w, g

    def snapshot(self):
        with self.lock:                               # keep the lock short: the game loop adds rows through it
            n = self.n
            X16, Y, W, G = self.X[:n].copy(), self.Y[:n].copy(), self.W[:n].copy(), self.G[:n].copy()
        return X16.astype(np.float32), Y, W, G

    def save(self, path: str) -> None:
        X, Y, W, G = self.snapshot()
        np.savez_compressed(path, X=X.astype(np.float16), Y=Y, W=W, G=G, groups=self.groups)

    @classmethod
    def load(cls, path: str, n_features: int, max_rows: int = 30000) -> "ExperienceMemory":
        mem = cls(n_features, max_rows)
        with np.load(path) as z:
            X, Y, W, G = z["X"], z["Y"], z["W"], z["G"]
            if X.shape[1] != n_features:
                raise ValueError(f"saved memory has {X.shape[1]} features, this brain has {n_features}")
            for i in range(min(len(X), max_rows)):
                mem.add(X[i], Y[i], W[i], int(G[i]))
            mem.groups = int(z["groups"])
        return mem


# ====================================================================== 2. hindsight labelling
@dataclass
class LabelConfig:
    window_s: float = 1.0          # how far back an outcome looks
    stride: int = 2                # keep every n-th tick of a window (press ticks are always kept)
    lead_shield: float = 0.30      # seconds before impact the guard has to be up
    tol_shield: float = 0.45       # ... and it stays up until impact
    lead_parry: float = 0.12       # seconds before impact the parry press is effective
    tol_parry: float = 0.07
    lead_jump: float = 0.28
    tol_jump: float = 0.10
    lead_move: float = 0.40        # seconds before impact the evasive step has to start
    tol_move: float = 0.35
    credit_tol: float = 0.06       # ticks this close to a successful press count as the press
    w_cf: float = 0.5              # weight of hindsight counterfactuals relative to real successes
    w_neg: float = 0.15            # weight of "not yet / too late" negatives inside a window
    w_quiet: float = 0.3
    quiet_stride: int = 8
    threat_quiet: float = 0.15     # below this imminence a tick counts as quiet
    shield_arc_deg: float = 100.0  # the guard only helps against hits from this arc in front
    dmg_ref: float = 0.1           # damage as a fraction of max HP that counts as one unit


# outcome -> (heads credited, base weight)
SUCCESS_SPECS: Dict[int, Tuple[Tuple[int, ...], float]] = {
    EventKind.PARRY:         ((PARRY, SHIELD, MOVE_X, MOVE_Z), 1.0),
    EventKind.PERFECT_DODGE: ((JUMP, MOVE_X, MOVE_Z), 1.0),
    EventKind.BLOCK:         ((SHIELD,), 0.6),
    EventKind.DODGE:         ((MOVE_X, MOVE_Z, JUMP), 0.5),
    EventKind.HIT_LANDED:    ((ATTACK, MOVE_X, MOVE_Z), 0.5),
    EventKind.KILL:          ((ATTACK, MOVE_X, MOVE_Z), 0.8),
}


@dataclass
class TickRecord:
    t: float
    x: np.ndarray                  # trace features at this tick
    a: np.ndarray                  # action actually sent (N_HEADS,)
    threat: float = 0.0
    bearing: float = NAN
    evade: Optional[np.ndarray] = None
    consumed: bool = False
    idx: int = 0


class HindsightLabeller:
    def __init__(self, memory: ExperienceMemory, cfg: Optional[LabelConfig] = None):
        self.mem = memory
        self.cfg = cfg or LabelConfig()
        c = self.cfg
        self._lt = {SHIELD: (c.lead_shield, c.tol_shield), PARRY: (c.lead_parry, c.tol_parry),
                    JUMP: (c.lead_jump, c.tol_jump), MOVE_X: (c.lead_move, c.tol_move),
                    MOVE_Z: (c.lead_move, c.tol_move)}
        self.pending: collections.deque = collections.deque()
        self.stats: collections.Counter = collections.Counter()
        self.outcomes_since_fit = 0
        self._tick = 0

    # ------------------------------------------------------------------ per tick
    def push(self, rec: TickRecord) -> None:
        if self.pending and rec.t < self.pending[-1].t - 1e-6:    # clock went backwards: pause, reload
            self.pending.clear()
        rec.idx = self._tick
        self._tick += 1
        self.pending.append(rec)
        horizon = self.cfg.window_s + 0.25
        while self.pending and rec.t - self.pending[0].t > horizon:
            old = self.pending.popleft()
            if (not old.consumed and old.threat < self.cfg.threat_quiet
                    and old.idx % self.cfg.quiet_stride == 0):
                w = np.full(N_HEADS, self.cfg.w_quiet, np.float32)
                self.mem.add(old.x, np.zeros(N_HEADS, np.float32), w, self.mem.groups)

    # ------------------------------------------------------------------ per outcome
    def resolve(self, ev: GameEvent, hp_max: float = 1.0) -> int:
        """Label the ticks before an outcome.  Returns the number of rows added to memory."""
        kind = int(ev.kind)
        self.stats[EventKind(kind).name] += 1
        T = ev.t
        rows = [r for r in self.pending if T - self.cfg.window_s <= r.t < T - 1e-6]
        if len(rows) < 4:
            return 0
        group = self.mem.new_group()
        if kind in SUCCESS_SPECS:
            n = self._success(rows, T, kind, group)
        elif kind == EventKind.HIT_TAKEN:
            n = self._failure(rows, T, ev, hp_max, group)
        elif kind == EventKind.THREAT_EXPIRED:
            n = self._neutral(rows, T, group)
        else:
            return 0
        for r in rows:
            r.consumed = True
        self.outcomes_since_fit += 1
        return n

    # ------------------------------------------------------------------ rule families
    def _keep(self, i: int, r: TickRecord, tau: float) -> bool:
        return i % self.cfg.stride == 0 or tau < 0.2 or bool(r.a[PARRY:].max() >= 0.5)

    def _commit(self, r: TickRecord, y: np.ndarray, w: np.ndarray, group: int) -> int:
        if not w.any():
            return 0
        self.mem.add(r.x, y, w, group)
        return 1

    def _success(self, rows: List[TickRecord], T: float, kind: int, group: int) -> int:
        c = self.cfg
        heads, base = SUCCESS_SPECS[kind]
        press = {}
        for h in PULSE_HEADS:                         # presses early enough to have caused this outcome
            lead, tol = self._lt.get(h, (0.5, 0.2))
            press[h] = np.array([r.t for r in rows if r.a[h] >= 0.5 and 0.0 < T - r.t <= lead + 2 * tol + 0.12])
        n = 0
        for i, r in enumerate(rows):
            tau = T - r.t
            if not self._keep(i, r, tau):
                continue
            y = np.zeros(N_HEADS, np.float32)
            w = np.zeros(N_HEADS, np.float32)
            for h in heads:
                if h in MOVE_HEADS:
                    y[h], w[h] = r.a[h], 0.5 * base
                elif h == SHIELD:
                    lead, tol = self._lt[SHIELD]
                    if tau <= lead + tol:
                        held = kind == EventKind.PARRY or r.a[SHIELD] >= 0.5   # a parry implies the guard was up
                        y[h], w[h] = (1.0, 0.6 * base) if held else (0.0, 0.3 * base)
                    else:
                        y[h], w[h] = 0.0, c.w_neg
                else:
                    ts = press[h]
                    if not ts.size:
                        continue                       # nothing pressed: this head is not responsible
                    dist = float(np.abs(ts - r.t).min())
                    if dist <= c.credit_tol:
                        y[h], w[h] = 1.0, 0.6 * base
                    elif dist > 2 * c.credit_tol:
                        y[h], w[h] = 0.0, c.w_neg
            n += self._commit(r, y, w, group)
        return n

    def _failure(self, rows: List[TickRecord], T: float, ev: GameEvent, hp_max: float, group: int) -> int:
        c = self.cfg
        scale = float(np.clip(0.5 + ev.amount / max(c.dmg_ref * hp_max, 1e-6), 0.5, 2.0)) if ev.amount > 0 else 1.0
        bearing = ev.bearing
        if not math.isfinite(bearing):                # fall back to what the eyes saw last
            for r in reversed(rows):
                if math.isfinite(r.bearing):
                    bearing = r.bearing
                    break
        frontal = (not math.isfinite(bearing)) or abs(bearing) <= math.radians(c.shield_arc_deg)
        heads = [JUMP, MOVE_X, MOVE_Z] + ([SHIELD, PARRY] if frontal else [])
        lead_s, tol_s = self._lt[SHIELD]
        tail = [r.a[SHIELD] for r in rows if T - r.t <= lead_s + tol_s]
        if SHIELD in heads and tail and float(np.mean(tail)) > 0.5:
            heads.remove(SHIELD)                      # the guard was already up and the hit still landed
        n = 0
        for i, r in enumerate(rows):
            tau = T - r.t
            if not self._keep(i, r, tau):
                continue
            y = np.zeros(N_HEADS, np.float32)
            w = np.zeros(N_HEADS, np.float32)
            for h in heads:
                lead, tol = self._lt[h]
                if h == SHIELD:
                    if tau <= lead + tol:
                        y[h], w[h] = 1.0, c.w_cf * scale
                    else:
                        y[h], w[h] = 0.0, c.w_neg
                elif h in (PARRY, JUMP):
                    d = abs(tau - lead)
                    if d <= tol:
                        y[h], w[h] = 1.0, c.w_cf * scale
                    elif d >= 2 * tol:
                        y[h], w[h] = 0.0, c.w_neg
                else:                                  # evasive step
                    if tau > lead + tol:
                        y[h], w[h] = 0.0, c.w_neg
                    elif r.evade is not None:
                        y[h], w[h] = float(r.evade[h - MOVE_X]), c.w_cf * scale
            n += self._commit(r, y, w, group)
        return n

    def _neutral(self, rows: List[TickRecord], T: float, group: int) -> int:
        n = 0
        for i, r in enumerate(rows):
            if self._keep(i, r, T - r.t):
                n += self._commit(r, np.zeros(N_HEADS, np.float32), np.full(N_HEADS, 2 * self.cfg.w_neg, np.float32), group)
        return n


# ====================================================================== 4. linear readout
@dataclass
class ReadoutConfig:
    ranks: Tuple[int, ...] = (5, 20, 60)            # PCA ranks tried (as in the fly.ai SSH Fighter readout)
    alphas: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    folds: int = 5
    pca_rows: int = 6000
    min_rows: int = 300
    min_groups: int = 8
    min_pos: int = 15
    useful_r2: float = 0.05                         # continuous heads: held-out R2 at least this
    useful_auc: float = 0.62                        # binary heads: held-out AUC at least this
    seed: int = 0


def randomized_pca(Xc: np.ndarray, k: int, seed: int = 0, n_iter: int = 3, oversample: int = 10) -> np.ndarray:
    """Top-k principal directions (d x k) of centred data, without forming the d x d covariance."""
    n, d = Xc.shape
    r = min(k + oversample, d, n)
    Q = np.random.default_rng(seed).standard_normal((d, r)).astype(np.float32)
    Y = Xc @ Q
    for _ in range(n_iter):
        Y, _ = np.linalg.qr(Y)
        Z, _ = np.linalg.qr(Xc.T @ Y)
        Y = Xc @ Z
    Y, _ = np.linalg.qr(Y)
    _, _, Vt = np.linalg.svd(Y.T @ Xc, full_matrices=False)
    return Vt[:k].T.astype(np.float32)


def _moments(Z: np.ndarray, y: np.ndarray, w: np.ndarray):
    wz = Z * w[:, None]
    return (float(w.sum()), wz.sum(0), float((w * y).sum()), Z.T @ wz, wz.T @ y)


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _solve(mom, k: int, alpha: float):
    Sw, Sz, Sy, Szz, Szy = mom
    mz, my = Sz[:k] / Sw, Sy / Sw
    C = Szz[:k, :k] / Sw - np.outer(mz, mz)
    r = Szy[:k] / Sw - mz * my
    beta = np.linalg.solve(C + alpha * np.eye(k), r)
    return beta, float(my - mz @ beta)


def weighted_r2(y, p, w) -> float:
    ybar = float((w * y).sum() / w.sum())
    den = float((w * (y - ybar) ** 2).sum())
    return float("nan") if den <= 1e-12 else 1.0 - float((w * (y - p) ** 2).sum()) / den


def auc_score(pos: np.ndarray, p: np.ndarray) -> float:
    pos = pos.astype(bool)
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]          # average rank for ties
    return float((ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def best_threshold(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    """Threshold on held-out scores that maximises weighted balanced accuracy."""
    pos = y > 0.5
    wp, wn = float(w[pos].sum()), float(w[~pos].sum())
    if wp <= 0 or wn <= 0:
        return 0.5
    cands = np.unique(np.quantile(p, np.linspace(0.3, 0.995, 80)))
    best, thr = -1.0, float(cands[len(cands) // 2])
    for t in cands:
        hit = p >= t
        bal = 0.5 * (float(w[pos & hit].sum()) / wp + float(w[~pos & ~hit].sum()) / wn)
        if bal > best:
            best, thr = bal, float(t)
    return thr


@dataclass
class HeadReport:
    head: str
    rows: int = 0
    pos: int = 0
    k: int = 0
    alpha: float = 0.0
    r2: float = NAN
    auc: float = NAN
    thr: float = NAN
    useful: bool = False


class LinearReadout:
    """scores = x @ W + b, one column per head.  PCA and standardisation are folded into W and b."""

    def __init__(self, W, b, thr, useful, report: List[HeadReport]):
        self.W = np.asarray(W, np.float32)
        self.b = np.asarray(b, np.float32)
        self.thr = np.asarray(thr, np.float32)
        self.useful = np.asarray(useful, bool)
        self.report = report

    def scores(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b

    def format_report(self) -> str:
        lines = [f"{'head':<7} {'rows':>6} {'pos':>5} {'rank':>4} {'alpha':>6} {'R2':>7} {'AUC':>6} {'thr':>7}  useful"]
        for r in self.report:
            lines.append(f"{r.head:<7} {r.rows:>6} {r.pos:>5} {r.k:>4} {r.alpha:>6g} {r.r2:>7.3f} {r.auc:>6.3f} "
                         f"{r.thr:>7.3f}  {'yes' if r.useful else 'no'}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray, Wt: np.ndarray, G: np.ndarray,
            cfg: Optional[ReadoutConfig] = None) -> "LinearReadout":
        cfg = cfg or ReadoutConfig()
        X = np.asarray(X, np.float32)
        n, d = X.shape
        rng = np.random.default_rng(cfg.seed)
        mu = X.mean(0)
        sd = X.std(0)
        sd[sd < 1e-3] = 1.0                                       # silent neurons carry nothing
        Xz = (X - mu) / sd
        kmax = int(min(max(cfg.ranks), d, max(n - 1, 1)))
        sub = rng.choice(n, min(n, cfg.pca_rows), replace=False)
        V = randomized_pca(Xz[sub] - Xz[sub].mean(0), kmax, cfg.seed)
        Z = (Xz @ V).astype(np.float64)
        fold = (np.asarray(G) % cfg.folds).astype(int)

        Wfull = np.zeros((d, N_HEADS), np.float32)
        bias = np.zeros(N_HEADS, np.float32)
        thr = np.full(N_HEADS, 0.5, np.float32)
        useful = np.zeros(N_HEADS, bool)
        report: List[HeadReport] = []

        for h in range(N_HEADS):
            m = Wt[:, h] > 0
            y_h, w_h = Y[m, h].astype(np.float64), Wt[m, h].astype(np.float64)
            Zm, fm, gm = Z[m], fold[m], np.asarray(G)[m]
            rep = HeadReport(HEADS[h], rows=int(m.sum()),
                             pos=int((y_h > 0.5).sum()) if IS_BINARY[h] else int((np.abs(y_h) > 0.2).sum()))
            report.append(rep)
            if rep.rows < cfg.min_rows or len(np.unique(gm)) < cfg.min_groups or rep.pos < 2:
                continue
            total = _moments(Zm, y_h, w_h)
            fold_mom = {f: _moments(Zm[fm == f], y_h[fm == f], w_h[fm == f]) for f in range(cfg.folds) if (fm == f).any()}
            best = (-np.inf, None, None, None)
            for k in [r for r in cfg.ranks if r <= kmax] or [kmax]:
                for alpha in cfg.alphas:
                    oof = np.full(len(y_h), np.nan)
                    for f, mom_f in fold_mom.items():
                        mom_tr = _sub(total, mom_f)
                        if mom_tr[0] <= 1e-9:
                            continue
                        beta, c0 = _solve(mom_tr, k, alpha)
                        te = fm == f
                        oof[te] = Zm[te, :k] @ beta + c0
                    ok = np.isfinite(oof)
                    if ok.sum() < 0.5 * len(y_h):
                        continue
                    r2 = weighted_r2(y_h[ok], oof[ok], w_h[ok])
                    if np.isfinite(r2) and r2 > best[0]:
                        best = (r2, k, alpha, oof.copy())
            r2, k, alpha, oof = best
            if k is None:
                continue
            beta, c0 = _solve(total, k, alpha)
            q = V[:, :k] @ beta
            Wfull[:, h] = q / sd
            bias[h] = c0 - float(mu @ (q / sd))
            rep.k, rep.alpha, rep.r2 = k, alpha, float(r2)
            ok = np.isfinite(oof)
            if IS_BINARY[h]:
                rep.auc = auc_score(y_h[ok] > 0.5, oof[ok])
                thr[h] = best_threshold(y_h[ok], oof[ok], w_h[ok])
                rep.thr = float(thr[h])
                useful[h] = bool(rep.pos >= cfg.min_pos and np.isfinite(rep.auc) and rep.auc >= cfg.useful_auc)
            else:
                useful[h] = bool(np.isfinite(r2) and r2 >= cfg.useful_r2)
            rep.useful = bool(useful[h])
        return cls(Wfull, bias, thr, useful, report)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        np.savez_compressed(path, W=self.W, b=self.b, thr=self.thr, useful=self.useful,
                            rep=np.array([[r.rows, r.pos, r.k, r.alpha, r.r2, r.auc, r.thr] for r in self.report], float))

    @classmethod
    def load(cls, path: str) -> "LinearReadout":
        with np.load(path) as z:
            rep = [HeadReport(HEADS[i], int(r[0]), int(r[1]), int(r[2]), float(r[3]), float(r[4]), float(r[5]),
                              float(r[6]), bool(z["useful"][i])) for i, r in enumerate(z["rep"])]
            return cls(z["W"], z["b"], z["thr"], z["useful"], rep)


class AsyncFitter:
    """Refits in a background thread so the game loop never stalls; the new readout is swapped in
    by ``on_done``.  ``blocking=True`` fits inline (used by tests)."""

    def __init__(self, memory: ExperienceMemory, cfg: ReadoutConfig, on_done: Callable[[LinearReadout], None]):
        self.memory, self.cfg, self.on_done = memory, cfg, on_done
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None
        self.fits = 0

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def submit(self, blocking: bool = False) -> bool:
        if self.busy:
            return False
        X, Y, W, G = self.memory.snapshot()

        def work():
            try:
                self.on_done(LinearReadout.fit(X, Y, W, G, self.cfg))
                self.fits += 1
            except BaseException as exc:      # never let a fit take the game loop down
                self.error = exc

        if blocking:
            work()
        else:
            self.thread = threading.Thread(target=work, daemon=True, name="readout-fit")
            self.thread.start()
        return True


# ====================================================================== 5. decoder
@dataclass
class DecoderConfig:
    move_gain: float = 1.0
    move_deadzone: float = 0.15
    shield_off_ratio: float = 0.6      # shield drops when its score falls below this share of its threshold
    shield_min_hold: float = 0.25      # s
    cooldown_parry: float = 0.35       # s between presses
    cooldown_jump: float = 0.6
    cooldown_attack: float = 0.3
    jump_lockout: float = 0.4          # the guard stays down this long after a jump
    stamina_min_jump: float = 0.0


class ActionDecoder:
    """Scores -> controller state.  Heads whose held-out score is not useful yet stay neutral, so the
    fly does nothing rather than act on noise.  Body limits (airborne, stunned, no shield) are the
    only game knowledge used here."""

    def __init__(self, cfg: Optional[DecoderConfig] = None):
        self.cfg = cfg or DecoderConfig()
        self.seq = 0
        self._t_prev = -1e9
        self._reset_timers()

    def _reset_timers(self) -> None:
        self.shield_on = False
        self.shield_since = -1e9
        self.last_press = {PARRY: -1e9, JUMP: -1e9, ATTACK: -1e9}
        self.jump_until = -1e9

    def decide(self, scores: Optional[np.ndarray], readout: Optional[LinearReadout], link: LinkState,
               t: float) -> ActionState:
        c = self.cfg
        if t < self._t_prev - 1e-6:                   # game clock went backwards (reload, new session): old timers are meaningless
            self._reset_timers()
        self._t_prev = t
        act = ActionState(seq=self.seq)
        self.seq += 1
        if scores is None or readout is None or link.stunned:
            self.shield_on = False
            return act
        u, thr = readout.useful, readout.thr

        if u[MOVE_X] or u[MOVE_Z]:
            mx = float(scores[MOVE_X]) * c.move_gain if u[MOVE_X] else 0.0
            mz = float(scores[MOVE_Z]) * c.move_gain if u[MOVE_Z] else 0.0
            mag = math.hypot(mx, mz)
            if mag >= c.move_deadzone:
                s = min(1.0, mag) / mag
                act.move_x, act.move_z = mx * s, mz * s

        grounded = link.on_ground
        can_guard = link.has_shield and grounded and t >= self.jump_until

        def margin(h):
            return (float(scores[h]) - float(thr[h])) / max(abs(float(thr[h])), 1e-3)

        # guard with hysteresis
        if u[SHIELD] and can_guard:
            s, on_thr = float(scores[SHIELD]), float(thr[SHIELD])
            if not self.shield_on and s >= on_thr:
                self.shield_on, self.shield_since = True, t
            elif self.shield_on and s < on_thr * c.shield_off_ratio and t - self.shield_since >= c.shield_min_hold:
                self.shield_on = False
        else:
            self.shield_on = False

        # one-shot presses; parry and jump cannot both go out in the same tick
        want = {}
        if u[PARRY] and link.has_shield and grounded and scores[PARRY] >= thr[PARRY] \
                and t - self.last_press[PARRY] >= c.cooldown_parry:
            want[PARRY] = margin(PARRY)
        if u[JUMP] and grounded and scores[JUMP] >= thr[JUMP] and t - self.last_press[JUMP] >= c.cooldown_jump \
                and link.stamina >= c.stamina_min_jump:
            want[JUMP] = margin(JUMP)
        if want:
            h = max(want, key=want.get)
            self.last_press[h] = t
            if h == PARRY:
                act.parry = True
                self.shield_on = True
                self.shield_since = t
            else:
                act.jump = True
                self.shield_on = False
                self.jump_until = t + c.jump_lockout
        if u[ATTACK] and scores[ATTACK] >= thr[ATTACK] and t - self.last_press[ATTACK] >= c.cooldown_attack:
            act.attack = True
            self.last_press[ATTACK] = t
        act.shield = self.shield_on
        return act


# ====================================================================== exploration (training only)
@dataclass
class ExploreConfig:
    threat_min: float = 0.30       # a threat this imminent makes exploration eligible
    rate: float = 1.2              # defensive episodes per second while a threat is near
    attack_rate: float = 1.5
    approach_rate: float = 0.3
    attack_range: float = 2.6
    approach_min_dist: float = 4.0
    decay_outcomes: int = 600      # after this many outcomes exploration falls to min_scale
    min_scale: float = 0.25
    cooldown: float = 0.4


class Explorer:
    """Random actions at salient moments so that successes (a parry that really landed) can happen at
    all and be learned from.  The sensory trigger only says WHEN to try something; WHAT is random."""

    def __init__(self, cfg: Optional[ExploreConfig] = None, seed: int = 0):
        self.cfg = cfg or ExploreConfig()
        self.rng = np.random.default_rng(seed)
        self.cur: Optional[ActionState] = None      # a held action (shield, a step) that is still running
        self.until = -1e9
        self.next_ok = -1e9
        self._t_prev = -1e9

    def _start(self, t: float, act: ActionState, dur: float) -> ActionState:
        if dur <= 0.0:                               # one-shot press: sent now, exactly once
            self.cur = None
            self.next_ok = t + self.cfg.cooldown
        else:
            self.cur, self.until = act, t + dur
        return act

    def step(self, t: float, dt: float, p: Percept, n_outcomes: int) -> Optional[ActionState]:
        c = self.cfg
        if t < self._t_prev - 1e-6:                   # game clock went backwards
            self.cur, self.until, self.next_ok = None, -1e9, -1e9
        self._t_prev = t
        if self.cur is not None:
            if t < self.until:
                return self.cur
            self.cur, self.next_ok = None, t + c.cooldown
        if t < self.next_ok:
            return None
        scale = max(c.min_scale, 1.0 - n_outcomes / max(c.decay_outcomes, 1))
        r = self.rng
        if p.threat >= c.threat_min:
            if r.random() < c.rate * scale * dt:
                can_hit = math.isfinite(p.enemy_dist) and p.enemy_dist <= c.attack_range and abs(p.enemy_az) < math.radians(60)
                kind = int(r.integers(0, 7 if can_hit else 6))
                if kind == 0:
                    return self._start(t, ActionState(shield=True), float(r.uniform(0.4, 0.9)))
                if kind == 1:
                    return self._start(t, ActionState(parry=True, shield=True), 0.0)
                if kind == 2:
                    return self._start(t, ActionState(jump=True), 0.0)
                if kind in (3, 4):
                    return self._start(t, ActionState(move_x=-1.0 if kind == 3 else 1.0), float(r.uniform(0.3, 0.6)))
                if kind == 5:
                    return self._start(t, ActionState(move_z=-1.0), 0.4)
                return self._start(t, ActionState(attack=True), 0.0)      # swing at an enemy that is in reach
        elif math.isfinite(p.enemy_dist):
            if p.enemy_dist <= c.attack_range and abs(p.enemy_az) < math.radians(60) and r.random() < c.attack_rate * scale * dt:
                return self._start(t, ActionState(attack=True), 0.0)
            if p.enemy_dist >= c.approach_min_dist and r.random() < c.approach_rate * scale * dt:
                return self._start(t, ActionState(move_x=math.sin(p.enemy_az), move_z=math.cos(p.enemy_az)), 0.5)
        return None
