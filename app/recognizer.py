"""Turning anonymous diarization labels ("Speaker 1") into names.

Open-set identification: for each anonymous speaker we compare their voice
fingerprint with the tenant's enrolled profiles. A name is only shown when the
best score clears the threshold; otherwise the speaker stays unnamed. Showing
the wrong colleague's name is worse than showing "Speaker 2", so the threshold
is tuned for a low false-accept rate (see scripts/evaluate.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .engine import ROOT

DEFAULT_THRESHOLD = 0.55
# Below this much speech a fingerprint is too noisy to trust.
MIN_CLUSTER_SECONDS = 3.0
# Extra margin for naming a second cluster after an already-named person.
SPLIT_MARGIN = 0.15


def calibrated_threshold() -> float:
    f = ROOT / "data" / "eval" / "metrics.json"
    if f.exists():
        return json.loads(f.read_text())["chosen_threshold"]
    return DEFAULT_THRESHOLD


@dataclass
class Match:
    cluster: int
    profile_id: str | None
    name: str | None
    score: float


def identify(cluster_vecs: dict[int, np.ndarray], cluster_seconds: dict[int, float],
             ids: list[str], names: list[str], profiles: np.ndarray,
             threshold: float) -> dict[int, Match]:
    """Greedy assignment from the highest score down: each person first gets
    their best-matching cluster (one-to-one), then split clusters may join."""
    result = {c: Match(c, None, None, 0.0) for c in cluster_vecs}
    if not ids:
        return result
    candidates = []
    for c, vec in cluster_vecs.items():
        scores = profiles @ vec
        result[c].score = float(scores.max())
        if cluster_seconds[c] < MIN_CLUSTER_SECONDS:
            continue
        candidates += [(float(s), c, i) for i, s in enumerate(scores) if s >= threshold]
    taken_clusters, taken_profiles = set(), set()
    for score, c, i in sorted(candidates, reverse=True):
        if c in taken_clusters or i in taken_profiles:
            continue
        result[c] = Match(c, ids[i], names[i], score)
        taken_clusters.add(c)
        taken_profiles.add(i)
    # Diarization sometimes splits one person into two clusters. A leftover
    # cluster may join an already-named person, but only with a stricter
    # score, because now it competes with nobody.
    for score, c, i in sorted(candidates, reverse=True):
        if c not in taken_clusters and score >= threshold + SPLIT_MARGIN:
            result[c] = Match(c, ids[i], names[i], score)
            taken_clusters.add(c)
    return result
