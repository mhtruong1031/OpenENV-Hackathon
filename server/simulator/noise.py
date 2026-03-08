"""Stochastic noise models for the biological simulator."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# Housekeeping and commonly detected human genes for realistic false-positive
# sampling (agent cannot distinguish true vs. false hits by name format alone).
_BACKGROUND_GENES: List[str] = [
    "ACTB", "GAPDH", "B2M", "RPL13A", "RPS18", "EEF1A1", "HSP90AB1", "HSPA8",
    "VIM", "TUBA1B", "FTL", "FTH1", "S100A4", "S100A6", "LGALS1", "ANXA2",
    "FLNA", "YBX1", "NPM1", "PCNA", "TMSB4X", "CD63", "CD81", "CD9", "ALB",
    "TUBB", "TUBB4B", "RPLP0", "RPLP1", "RPS3", "RPS4X", "RPS5", "RPS6",
    "RPS7", "RPS8", "RPS9", "RPS10", "RPS11", "RPS12", "RPS13", "RPS14",
    "RPS15", "RPS16", "RPS17", "RPS19", "RPS20", "RPS21", "RPS23", "RPS24",
    "RPS25", "RPS26", "RPS27", "RPS27A", "RPS28", "RPS29", "RPL3", "RPL4",
    "RPL5", "RPL6", "RPL7", "RPL7A", "RPL8", "RPL9", "RPL10", "RPL10A",
    "RPL11", "RPL12", "RPL14", "RPL15", "RPL17", "RPL18", "RPL18A", "RPL19",
    "RPL21", "RPL22", "RPL23", "RPL23A", "RPL24", "RPL26", "RPL27", "RPL28",
    "RPL29", "RPL30", "RPL31", "RPL32", "RPL34", "RPL35", "RPL35A", "RPL36",
    "RPL36AL", "RPL37", "RPL37A", "RPL38", "RPL39", "RPL40", "RPL41",
    "EEF1B2", "EEF1D", "EEF2", "HSP90AA1", "HSPA1A", "HSPA1B", "HSPA5",
    "HSPA9", "HSPB1", "HSPD1", "DNAJA1", "DNAJB1", "PPIA", "PPIB", "PPIG",
    "CST3", "CSTB", "SRP14", "SRP68", "SRP72", "ATP5F1A", "ATP5F1B", "ATP5F1C",
    "COX4I1", "COX5A", "NDUFA4", "NDUFB2", "UBC", "UBA52", "RPS3A", "RACK1",
    "LDHA", "PKM", "ENO1", "PGK1", "TPI1", "GPI", "ALDOA", "GAPDH", "PGAM1",
    "LDHB", "PFKL", "SLC25A5", "SLC25A6", "VDAC1", "VDAC2", "CYCS", "COX7A2",
    "ATP6V1A", "ATP6V1B2", "ATP6V1D", "ATP6V1E1", "ATP6V0D1", "ATP6V0E1",
]


class NoiseModel:
    """Generates calibrated noise for simulated experimental outputs.

    All randomness is funnelled through a single ``numpy.Generator``
    so that episodes are reproducible given the same seed.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    # ── expression-level noise ──────────────────────────────────────────

    def add_expression_noise(
        self,
        true_values: Dict[str, float],
        noise_level: float,
        dropout_rate: float,
    ) -> Dict[str, float]:
        noisy: Dict[str, float] = {}
        for gene, value in true_values.items():
            # Dropout probability is inversely proportional to expression
            # magnitude: lowly expressed genes drop out much more readily,
            # matching the zero-inflation pattern in real scRNA-seq data.
            p_drop = dropout_rate / (1.0 + abs(value))
            if self.rng.random() < p_drop:
                noisy[gene] = 0.0
            else:
                sigma = noise_level * abs(value) + 0.1
                noisy[gene] = float(value + self.rng.normal(0, sigma))
        return noisy

    # ── effect-size sampling ────────────────────────────────────────────

    def sample_effect_sizes(
        self,
        true_effects: Dict[str, float],
        sample_size: int,
        noise_level: float,
    ) -> Dict[str, float]:
        se = noise_level / max(np.sqrt(max(sample_size, 1)), 1e-6)
        return {
            gene: float(effect + self.rng.normal(0, se))
            for gene, effect in true_effects.items()
        }

    def sample_p_values(
        self,
        true_effects: Dict[str, float],
        sample_size: int,
        noise_level: float,
    ) -> Dict[str, float]:
        """Simulate approximate p-values from z-statistics."""
        from scipy import stats  # type: ignore[import-untyped]

        p_values: Dict[str, float] = {}
        se = noise_level / max(np.sqrt(max(sample_size, 1)), 1e-6)
        for gene, effect in true_effects.items():
            z = abs(effect) / max(se, 1e-8)
            p_values[gene] = float(2 * stats.norm.sf(z))
        return p_values

    # ── false discovery helpers ─────────────────────────────────────────

    def generate_false_positives(
        self, n_background_genes: int, fdr: float
    ) -> List[str]:
        n_fp = int(self.rng.binomial(n_background_genes, fdr))
        n_sample = min(n_fp, len(_BACKGROUND_GENES))
        if n_sample <= 0:
            return []
        indices = self.rng.choice(
            len(_BACKGROUND_GENES), size=n_sample, replace=False
        )
        return [_BACKGROUND_GENES[int(i)] for i in indices]

    def generate_false_negatives(
        self, true_genes: List[str], fnr: float
    ) -> List[str]:
        """Return the subset of *true_genes* that are missed."""
        return [g for g in true_genes if self.rng.random() < fnr]

    # ── quality helpers ─────────────────────────────────────────────────

    def quality_degradation(
        self, base_quality: float, factors: List[float]
    ) -> float:
        q = base_quality
        for f in factors:
            q *= f
        return float(np.clip(q + self.rng.normal(0, 0.02), 0.0, 1.0))

    def sample_qc_metric(
        self, mean: float, std: float, clip_lo: float = 0.0, clip_hi: float = 1.0
    ) -> float:
        return float(np.clip(self.rng.normal(mean, std), clip_lo, clip_hi))

    def sample_count(self, lam: float) -> int:
        return int(self.rng.poisson(max(lam, 0)))

    def coin_flip(self, p: float) -> bool:
        return bool(self.rng.random() < p)

    def sample_cluster_count(
        self, n_true_populations: int, quality: float
    ) -> int:
        """Over- or under-clustering depending on preprocessing quality."""
        delta = self.rng.integers(-2, 3)
        noise_clusters = max(0, int(round((1.0 - quality) * 3)))
        return max(1, n_true_populations + delta + noise_clusters)

    def shuffle_ranking(
        self, items: List[str], noise_level: float
    ) -> List[str]:
        """Permute a ranking with Gaussian noise on ordinals."""
        n = len(items)
        if n == 0:
            return []
        scores = np.arange(n, dtype=float) + self.rng.normal(
            0, noise_level * n, size=n
        )
        order = np.argsort(scores)
        return [items[int(i)] for i in order]
