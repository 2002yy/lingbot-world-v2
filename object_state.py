#!/usr/bin/env python
"""§39: sparse gameplay object state -- a persistent truth layer that lives
independently of the world model's implicit memory.

Design constraints come from §38B-C:
  - within 1-3 s objects do NOT vanish and are still 'the same thing'
  - the dimension that degrades first is TEXTURE/appearance, then pixel position
  => do NOT lock down textures; do NOT build a dense visual registry or a scene
     graph. Keep a SPARSE gameplay registry of things whose state must survive
     longer than the model's implicit memory (T50 ~ 3.25 s).

    ObjectState
      persistent_id    unique across time
      semantic_type    door / box / switch / ...
      world_transform  position + rotation (+ scale if needed)
      bounds           rough extent / collision volume
      gameplay_state   open/closed, intact/broken, carried, ...
      persistence_meta last_seen, confidence, source, revision
      visual_anchor    OPTIONAL low-frequency identity feature (DINO embedding);
                       never a texture

§40D-D4 addition -- state_anchor:
  §40D-D3 measured cos(a_id, d_state) = -0.41 at the revisit, i.e. an additive
  "identity anchor + state delta" formulation makes the two fight each other.
  The natural fix is to stop treating state as an additive correction and let
  identity+state live in ONE anchor:

      gameplay_state = OPEN  ->  active_anchor = door_OPEN

  so resurrection IS the state, with no separate KV impulse. The store keeps a
  small dict of state-specific anchors and exposes the one matching the current
  gameplay_state. Only the ACTIVE anchor is used at runtime.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ObjectState:
    persistent_id: int
    semantic_type: str
    world_transform: np.ndarray          # 4x4
    bounds: np.ndarray                   # [3] extents
    gameplay_state: dict = field(default_factory=dict)
    last_seen: float = 0.0
    confidence: float = 1.0
    source: str = "init"
    revision: int = 0
    visual_anchor: Optional[np.ndarray] = None   # low-freq identity only
    state_anchors: dict = field(default_factory=dict)
    #   state_key -> latent patch, e.g. {"closed": z_closed, "open": z_open}
    active_anchor_key: Optional[str] = None

    # ---- §41B motion: transform(t) is authoritative and updated per chunk ----
    velocity: Optional[np.ndarray] = None            # [3] world units / s
    angular_velocity: Optional[np.ndarray] = None    # [3] rad / s
    # NOTE: motion is the AUTHORITATIVE simulation output. The renderer only
    # projects it; the refresh cadence of the render binding is a separate
    # concern (see refresh policy).

    # ---- §41B-2b prediction vs observation (kept strictly separate) ----
    uv_pred: Optional[Tuple[float, float]] = None       # project(transform(t))
    uv_measured: Optional[Tuple[float, float]] = None   # visual reacquisition
    reprojection_error: Optional[float] = None          # measured - pred
    # The measured position is a DIAGNOSTIC. It is never written back into
    # world_transform, otherwise physics would be dragged around by the
    # jitter of the generated picture.

    @property
    def active_anchor(self):
        """The anchor matching the current gameplay_state (§40D-D4)."""
        if self.active_anchor_key is not None:
            return self.state_anchors.get(self.active_anchor_key)
        return self.visual_anchor

    def anchor_for_state(self, state_key: str):
        return self.state_anchors.get(state_key)

    def touch(self, t: float, confidence: float, source: str):
        self.last_seen = t
        self.confidence = float(confidence)
        self.source = source
        self.revision += 1


class PersistentObjectStore:
    """Sparse registry: only objects whose state must outlive the model's memory."""

    MATCH_ACCEPT = 0.45      # min anchor similarity to accept a reacquisition
    MATCH_MARGIN = 0.05      # best must beat the runner-up by this much

    def __init__(self):
        self._objs: Dict[int, ObjectState] = {}
        self._next_id = 1
        self.log: List[dict] = []

    # ---------- write ----------
    def register(self, semantic_type, world_transform, bounds,
                 anchor=None, t=0.0, gameplay_state=None) -> ObjectState:
        oid = self._next_id
        self._next_id += 1
        st = ObjectState(
            persistent_id=oid, semantic_type=semantic_type,
            world_transform=np.asarray(world_transform, float).copy(),
            bounds=np.asarray(bounds, float).copy(),
            gameplay_state=dict(gameplay_state or {}),
            last_seen=t, confidence=1.0, source="register",
            visual_anchor=None if anchor is None else np.asarray(anchor, float).copy())
        self._objs[oid] = st
        self.log.append(dict(event="register", id=oid, type=semantic_type, t=t))
        return st

    def update(self, oid, world_transform=None, bounds=None, anchor=None,
               t=None, confidence=None, gameplay_state=None):
        st = self._objs[oid]
        if world_transform is not None:
            st.world_transform = np.asarray(world_transform, float).copy()
        if bounds is not None:
            st.bounds = np.asarray(bounds, float).copy()
        if anchor is not None:
            st.visual_anchor = np.asarray(anchor, float).copy()
        if gameplay_state:
            st.gameplay_state.update(gameplay_state)
        if t is not None:
            st.touch(t, confidence if confidence is not None else st.confidence,
                     "reacquire")
        return st

    def set_state(self, oid, **kv):
        """§39C: the authoritative gameplay truth (independent of rendering).

        §40D-D4: if the new gameplay_state maps to a known state anchor, the
        object's active anchor switches to it -- resurrection then IS the
        state, with no additive state delta.
        """
        obj = self._objs[oid]
        obj.gameplay_state.update(kv)
        obj.revision += 1
        # state_key precedence: explicit "state_key" > open flag > first key
        skey = kv.get("state_key")
        if skey is None and "open" in kv:
            skey = "open" if kv["open"] else "closed"
        if skey is not None and skey in obj.state_anchors:
            obj.active_anchor_key = skey
        self.log.append(dict(event="set_state", id=oid, state=dict(kv),
                             active_anchor=obj.active_anchor_key))
        return obj

    def set_motion(self, oid, velocity=None, angular_velocity=None):
        """§41B: authoritative motion state (simulation output)."""
        st = self._objs[oid]
        if velocity is not None:
            st.velocity = np.asarray(velocity, float).copy()
        if angular_velocity is not None:
            st.angular_velocity = np.asarray(angular_velocity, float).copy()
        return st

    def integrate(self, oid, dt: float):
        """Advance transform by the authoritative motion (simple Euler)."""
        st = self._objs[oid]
        if st.velocity is not None:
            st.world_transform[:3, 3] = st.world_transform[:3, 3] + st.velocity * dt
        return st

    def set_anchor_for_state(self, oid, state_key, latent=None, feature=None):
        """Attach a state-specific prototype to an object.

        §40D-D5: a state prototype carries BOTH
            latent   -- the state-specific latent patch used for re-render
            feature  -- a low-frequency identity feature used for MATCHING
        Identity matching then takes the best score over ALL state prototypes,
        so a door rendered OPEN and a door rendered CLOSED are both recognised
        as door_01. State is judged separately against the authoritative
        gameplay_state.
        """
        obj = self._objs[oid]
        slot = obj.state_anchors.setdefault(state_key, {})
        if latent is not None:
            slot["latent"] = latent
        if feature is not None:
            slot["feature"] = np.asarray(feature, float)
        return obj

    def state_features(self, oid):
        """All identity features of an object, across every known state."""
        return [v["feature"] for v in self._objs[oid].state_anchors.values()
                if v.get("feature") is not None]

    def state_latent(self, oid, state_key):
        slot = self._objs[oid].state_anchors.get(state_key) or {}
        return slot.get("latent")

    def forget(self, oid):
        self._objs.pop(oid, None)

    # ---------- read ----------
    def __len__(self):
        return len(self._objs)

    def __iter__(self):
        return iter(self._objs.values())

    def get(self, oid) -> Optional[ObjectState]:
        return self._objs.get(oid)

    def anchors(self) -> Tuple[List[int], List[np.ndarray]]:
        ids, vecs = [], []
        for o in self._objs.values():
            if o.visual_anchor is not None:
                ids.append(o.persistent_id)
                vecs.append(o.visual_anchor)
        return ids, vecs

    def anchors_multi(self) -> Tuple[List[int], List[List[np.ndarray]]]:
        """§40D-D5: per-object list of identity prototypes over ALL states.

        Falls back to the single `visual_anchor` when no state prototypes
        have been registered.
        """
        ids, groups = [], []
        for o in self._objs.values():
            feats = [f for f in self.state_features(o.persistent_id)]
            if not feats and o.visual_anchor is not None:
                feats = [o.visual_anchor]
            if feats:
                ids.append(o.persistent_id)
                groups.append(feats)
        return ids, groups

    # ---------- reacquisition ----------
    def set_prediction(self, oid, uv=None):
        """§41B-2b: set the projected position of the authoritative transform.

        This is what the spatial prior must use for a moving object. Using the
        last SEEN position instead penalises the motion itself (measured: at
        v=2.0 the accumulated penalty reached ~3.1 and killed the match even
        though existence was 100%).
        """
        st = self._objs[oid]
        st.uv_pred = None if uv is None else (float(uv[0]), float(uv[1]))
        return st

    def set_measurement(self, oid, uv, confidence=None):
        """Record the visual measurement as a DIAGNOSTIC only (never written
        back into world_transform)."""
        st = self._objs[oid]
        if uv is not None:
            st.uv_measured = (float(uv[0]), float(uv[1]))
            if st.uv_pred is not None:
                st.reprojection_error = float(
                    np.hypot(uv[0] - st.uv_pred[0], uv[1] - st.uv_pred[1]))
        if confidence is not None:
            st.confidence = float(confidence)
        return st

    def reacquire(self, candidates: List[dict], t: float,
                  accept=None, margin=None, pos_sigma=0.15,
                  pos_lambda=0.5, motion_aware=True) -> List[dict]:
        """Match candidate detections to existing objects.

        Uses BOTH the low-frequency visual anchor AND a spatial prior: the
        anchor alone is not discriminative between similar-looking objects
        (brick wall vs brick wall), so a candidate's expected position gates
        which existing objects it may match.

        §40D-D5: identity is matched against ALL state prototypes of an object
        and takes the BEST one. A door drawn OPEN or CLOSED is therefore still
        recognised as door_01; the state itself is judged separately against
        the authoritative gameplay_state. (Matching only the active state
        would re-break identity whenever the model renders the wrong state.)

        candidates: [{'anchor': vec, 'uv': (u,v) normalized image position,
                      'world_transform': 4x4, 'bounds': [3]}]
        """
        accept = self.MATCH_ACCEPT if accept is None else accept
        margin = self.MATCH_MARGIN if margin is None else margin
        ids, groups = self.anchors_multi()
        # §41B-2b: the spatial gate must be relative to the PREDICTED position
        # of the authoritative transform, not to the last seen position. Only
        # fall back to last-seen when no prediction exists (tracking lost).
        uvs = []
        for i in ids:
            o = self._objs[i]
            if motion_aware and o.uv_pred is not None:
                uvs.append(o.uv_pred)
            else:
                uvs.append(o.gameplay_state.get("uv", None))
        out = []
        for c in candidates:
            if not ids:
                out.append(dict(decision="new", id=None, sim=0.0, margin=0.0, t=t))
                continue
            sims, best_state = [], []
            for grp, uv in zip(groups, uvs):
                ss = [float(np.dot(c["anchor"], v)) for v in grp]
                k = int(np.argmax(ss))
                s = ss[k]
                if uv is not None and c.get("uv") is not None:
                    d = float(np.hypot(uv[0] - c["uv"][0], uv[1] - c["uv"][1]))
                    s -= pos_lambda * (d / max(pos_sigma, 1e-6)) ** 2
                sims.append(s)
                best_state.append(k)
            sims = np.array(sims)
            order = np.argsort(-sims)
            best = int(order[0])
            second = float(sims[order[1]]) if len(order) > 1 else -1.0
            m = float(sims[best] - second)
            if sims[best] < accept:
                dec = "new"
            elif m < margin:
                dec = "ambiguous"
            else:
                dec = "reacquire"
                self.update(ids[best], world_transform=c["world_transform"],
                            bounds=c["bounds"], anchor=c["anchor"], t=t,
                            confidence=float(sims[best]))
                # §41B-2b: record the measurement as a diagnostic; do NOT let
                # it drive the authoritative transform.
                if c.get("uv") is not None:
                    self.set_measurement(ids[best], c["uv"])
                    if self._objs[ids[best]].uv_pred is None:
                        # no prediction available -> last-seen is the only ref
                        self._objs[ids[best]].gameplay_state["uv"] = c.get("uv")
            self.log.append(dict(event=dec, id=ids[best] if dec == "reacquire" else None,
                                 sim=float(sims[best]), margin=m, t=t))
            out.append(dict(decision=dec,
                            id=ids[best] if dec == "reacquire" else None,
                            sim=float(sims[best]), margin=m, t=t,
                            matched_proto=int(best_state[best])))
        return out


if __name__ == "__main__":
    # tiny self-test of the state layer (no world model involved)
    store = PersistentObjectStore()
    a = np.random.default_rng(0).normal(size=16); a /= np.linalg.norm(a)
    b = np.random.default_rng(1).normal(size=16); b /= np.linalg.norm(b)
    o1 = store.register("door", np.eye(4), [1, 2, 0.2], anchor=a, t=0.0,
                        gameplay_state=dict(open=False))
    o2 = store.register("box", np.eye(4), [0.5, 0.5, 0.5], anchor=b, t=0.0)
    print("registered:", len(store))
    # exact anchors -> both reacquired
    dec = store.reacquire([dict(anchor=a, world_transform=np.eye(4), bounds=[1, 2, 0.2]),
                           dict(anchor=b, world_transform=np.eye(4), bounds=[.5, .5, .5])],
                          t=5.0)
    print("exact:", [d["decision"] for d in dec])
    # drifted/noisy anchor -> still reacquired or ambiguous, never silently new
    n = a + 0.15 * np.random.default_rng(2).normal(size=16); n /= np.linalg.norm(n)
    dec = store.reacquire([dict(anchor=n, world_transform=np.eye(4), bounds=[1, 2, 0.2])],
                          t=10.0)
    print("noisy:", dec[0]["decision"], round(dec[0]["sim"], 3))
    store.set_state(o1.persistent_id, open=True)
    print("door state:", store.get(o1.persistent_id).gameplay_state)
