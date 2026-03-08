"""Generate simulated intermediate outputs conditioned on latent state."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from models import (
    ActionType,
    ExperimentAction,
    IntermediateOutput,
    OutputType,
)

from .latent_state import FullLatentState
from .noise import NoiseModel

# Pool of common transcription factors used to generate realistic false-positive
# regulators, so the agent cannot trivially distinguish true vs. false hits by
# gene-name format alone.
_NOISE_TFS: List[str] = [
    "NR3C1", "KLF4", "EGR1", "IRF1", "FOSL2", "JUN", "FOS", "ATF3",
    "NFKB1", "RELA", "SP1", "MYC", "MAX", "E2F1", "CTCF", "YY1",
    "TP53", "STAT5A", "SMAD3", "TCF7L2", "NFE2L2", "HIF1A", "CREB1",
]

# Plausible generic pathway names for realistic false-positive enrichment hits
# (excluded from sampling when they overlap true_pathways).
_BACKGROUND_PATHWAYS: List[str] = [
    "Ribosome", "Oxidative_phosphorylation", "Cell_cycle", "p53_signaling_pathway",
    "MAPK_signaling", "PI3K_Akt_signaling", "Glycolysis", "Fatty_acid_metabolism",
    "Citrate_cycle_TCA_cycle", "Pyruvate_metabolism", "Amino_sugar_metabolism",
    "Pentose_phosphate_pathway", "Starch_sucrose_metabolism", "Galactose_metabolism",
    "Fructose_mannose_metabolism", "Purine_metabolism", "Pyrimidine_metabolism",
    "Spliceosome", "Proteasome", "RNA_polymerase", "Basal_transcription_factors",
    "DNA_replication", "Mismatch_repair", "Base_excision_repair", "Nucleotide_excision_repair",
    "Homologous_recombination", "Non_homologous_end_joining", "Apoptosis",
    "NF_kappa_B_signaling", "TNF_signaling", "TGF_beta_signaling", "Wnt_signaling",
    "Hippo_signaling", "Hedgehog_signaling", "Notch_signaling", "JAK_STAT_signaling",
    "Calcium_signaling", "cAMP_signaling", "mTOR_signaling", "AMPK_signaling",
    "Hippo_signaling_pathway", "Focal_adhesion", "ECM_receptor_interaction",
    "Cell_adhesion_molecules", "Tight_junction", "Gap_junction", "Phagosome",
    "Lysosome", "Endocytosis", "Autophagy", "Ubiquitin_mediated_proteolysis",
    "Oxidative_stress_response", "Hypoxia_response", "Unfolded_protein_response",
    "Cholesterol_metabolism", "Steroid_biosynthesis", "Bile_acid_metabolism",
    "Drug_metabolism_cytochrome_P450", "Chemical_carcinogenesis", "Metabolism_of_xenobiotics",
    "Immune_response", "Inflammatory_response", "Complement_cascade",
    "Antigen_processing_presentation", "T_cell_receptor_signaling", "B_cell_receptor_signaling",
    "Cytoskeleton", "Regulation_of_actin_cytoskeleton",
    "mRNA_surveillance_pathway", "RNA_degradation", "RNA_transport",
]


class OutputGenerator:
    """Creates structured ``IntermediateOutput`` objects conditioned on the
    hidden latent state, the action taken, and a stochastic noise model.
    """

    def __init__(self, noise: NoiseModel):
        self.noise = noise

    def generate(
        self,
        action: ExperimentAction,
        state: FullLatentState,
        step_index: int,
    ) -> IntermediateOutput:
        handler = _HANDLERS.get(action.action_type, self._default)
        return handler(self, action, state, step_index)

    # ── wet-lab outputs ─────────────────────────────────────────────────

    def _collect_sample(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        n_samples = action.parameters.get("n_samples", 6)
        quality = self.noise.quality_degradation(
            s.technical.sample_quality, [s.technical.capture_efficiency]
        )
        return IntermediateOutput(
            output_type=OutputType.SAMPLE_COLLECTION_RESULT,
            step_index=idx,
            quality_score=quality,
            summary=f"Collected {n_samples} samples (quality={quality:.2f})",
            data={
                "n_samples": n_samples,
                "quality": quality,
                "organism": "human",
                "tissue": "blood",
            },
            artifacts_available=["raw_samples"],
        )

    def _select_cohort(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        criteria = action.parameters.get("criteria", {})
        n_selected = action.parameters.get("n_selected", 4)
        return IntermediateOutput(
            output_type=OutputType.COHORT_RESULT,
            step_index=idx,
            summary=f"Selected cohort of {n_selected} samples with criteria {criteria}",
            data={"n_selected": n_selected, "criteria": criteria},
            artifacts_available=["cohort_manifest"],
        )

    def _prepare_library(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        complexity = self.noise.quality_degradation(
            s.technical.library_complexity,
            [s.technical.sample_quality],
        )
        return IntermediateOutput(
            output_type=OutputType.LIBRARY_PREP_RESULT,
            step_index=idx,
            quality_score=complexity,
            summary=f"Library prepared (complexity={complexity:.2f})",
            data={
                "library_complexity": complexity,
                "method": action.method or "10x_chromium",
            },
            artifacts_available=["prepared_library"],
        )

    def _culture_cells(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        days = action.parameters.get("days", 7)
        # Viability decays with culture duration: each day adds ~0.5%
        # cumulative stress, reflecting senescence, media depletion, and
        # passaging artefacts common in primary cell cultures.
        decay = 0.005 * days
        viability = self.noise.sample_qc_metric(
            max(0.50, 0.95 - decay), 0.05, 0.30, 1.0
        )
        return IntermediateOutput(
            output_type=OutputType.CULTURE_RESULT,
            step_index=idx,
            quality_score=viability,
            summary=f"Cultured for {days}d, viability={viability:.2f}",
            data={"days": days, "viability": viability},
            artifacts_available=["cultured_cells"],
        )

    def _perturb_gene(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        """Genetic perturbation (CRISPR/RNAi): high on-target efficiency,
        binary effect, non-trivial off-target risk."""
        target = action.parameters.get("target", "unknown")
        efficiency = s.last_perturbation_efficiency if s.last_perturbation_efficiency is not None else self.noise.sample_qc_metric(0.80, 0.12, 0.0, 1.0)
        off_target_risk = self.noise.sample_qc_metric(0.10, 0.05, 0.0, 0.5)
        return IntermediateOutput(
            output_type=OutputType.PERTURBATION_RESULT,
            step_index=idx,
            quality_score=efficiency,
            summary=(
                f"Genetic perturbation of {target} "
                f"(efficiency={efficiency:.2f}, off-target risk={off_target_risk:.2f})"
            ),
            data={
                "target": target,
                "efficiency": efficiency,
                "type": action.action_type.value,
                "off_target_risk": off_target_risk,
            },
            artifacts_available=["perturbed_cells"],
        )

    def _perturb_compound(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        """Small-molecule perturbation: dose-dependent, partial on-target
        activity, systemic effects possible."""
        target = action.parameters.get("target", "unknown")
        dose_um = action.parameters.get("dose_uM", 1.0)
        efficiency = s.last_perturbation_efficiency if s.last_perturbation_efficiency is not None else self.noise.sample_qc_metric(0.70, 0.15, 0.0, 1.0)
        on_target_frac = self.noise.sample_qc_metric(0.75, 0.10, 0.0, 1.0)
        return IntermediateOutput(
            output_type=OutputType.PERTURBATION_RESULT,
            step_index=idx,
            quality_score=efficiency * on_target_frac,
            summary=(
                f"Compound perturbation targeting {target} at {dose_um} µM "
                f"(efficiency={efficiency:.2f}, on-target={on_target_frac:.2f})"
            ),
            data={
                "target": target,
                "efficiency": efficiency,
                "type": action.action_type.value,
                "dose_uM": dose_um,
                "on_target_fraction": on_target_frac,
            },
            artifacts_available=["perturbed_cells"],
        )

    def _sequence_cells(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        import math
        depth = s.technical.sequencing_depth_factor
        n_cells = s.progress.n_cells_sequenced or self.noise.sample_count(
            s.biology.n_true_cells * s.technical.capture_efficiency
        )
        # Gene detection saturates with sequencing depth: follows a
        # 1 - exp(-k) saturation curve, scaled by library complexity.
        max_genes = 20_000
        saturation_arg = depth * s.technical.library_complexity * 0.8
        n_genes = self.noise.sample_count(
            int(max_genes * (1.0 - math.exp(-saturation_arg)))
        )
        median_umi = self.noise.sample_count(int(3000 * depth))
        quality = self.noise.quality_degradation(
            s.technical.sample_quality,
            [s.technical.library_complexity, s.technical.capture_efficiency],
        )
        return IntermediateOutput(
            output_type=OutputType.SEQUENCING_RESULT,
            step_index=idx,
            quality_score=quality,
            summary=(
                f"Sequenced {n_cells} cells, {n_genes} genes detected, "
                f"median UMI={median_umi}"
            ),
            data={
                "n_cells": n_cells,
                "n_genes": n_genes,
                "median_umi": median_umi,
                "sequencing_saturation": self.noise.sample_qc_metric(0.7, 0.1),
            },
            artifacts_available=["raw_count_matrix"],
        )

    # ── computational outputs ───────────────────────────────────────────

    def _run_qc(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        doublet_frac = self.noise.sample_qc_metric(
            s.technical.doublet_rate, 0.01, 0.0, 0.2
        )
        # Mitochondrial fraction reflects cellular stress: activated,
        # inflammatory, or pro-fibrotic populations have elevated mito
        # transcription compared to quiescent/resting cells.
        _stressed_states = {"activated", "stressed", "pro-fibrotic", "inflammatory"}
        has_stressed_cells = any(
            p.state in _stressed_states for p in s.biology.cell_populations
        )
        # Means are kept close (0.09 vs 0.06) with a wider SD (0.03) so the
        # mito fraction is informative but not a near-perfect oracle for
        # stressed-cell presence.
        mito_mean = 0.09 if has_stressed_cells else 0.06
        mito_frac = self.noise.sample_qc_metric(mito_mean, 0.03, 0.0, 0.3)
        ambient_frac = self.noise.sample_qc_metric(
            s.technical.ambient_rna_fraction, 0.01, 0.0, 0.2
        )
        warnings: List[str] = []
        if doublet_frac > 0.08:
            warnings.append(f"High doublet rate ({doublet_frac:.1%})")
        if mito_frac > 0.1:
            warnings.append(f"High mitochondrial fraction ({mito_frac:.1%})")
        quality = 1.0 - (doublet_frac + mito_frac + ambient_frac)
        return IntermediateOutput(
            output_type=OutputType.QC_METRICS,
            step_index=idx,
            quality_score=max(0.0, quality),
            summary="QC metrics computed",
            data={
                "doublet_fraction": doublet_frac,
                "mitochondrial_fraction": mito_frac,
                "ambient_rna_fraction": ambient_frac,
                "median_genes_per_cell": self.noise.sample_count(2500),
                "median_umi_per_cell": self.noise.sample_count(8000),
            },
            warnings=warnings,
            artifacts_available=["qc_report"],
        )

    def _filter_data(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        retain_frac = s.last_retain_frac if s.last_retain_frac is not None else self.noise.sample_qc_metric(0.85, 0.05, 0.5, 1.0)
        n_before = s.progress.n_cells_sequenced or s.biology.n_true_cells
        n_after = s.progress.n_cells_after_filter or max(100, int(n_before * retain_frac))
        return IntermediateOutput(
            output_type=OutputType.COUNT_MATRIX_SUMMARY,
            step_index=idx,
            quality_score=retain_frac,
            summary=f"Filtered {n_before} → {n_after} cells ({retain_frac:.0%} retained)",
            data={
                "n_cells_before": n_before,
                "n_cells_after": n_after,
                "n_genes_retained": self.noise.sample_count(15_000),
                "retain_fraction": retain_frac,
            },
            artifacts_available=["filtered_count_matrix"],
        )

    def _normalize_data(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        method = action.method or "log_normalize"
        return IntermediateOutput(
            output_type=OutputType.COUNT_MATRIX_SUMMARY,
            step_index=idx,
            summary=f"Normalized with {method}",
            data={"method": method, "n_hvg": self.noise.sample_count(2000)},
            artifacts_available=["normalized_matrix", "hvg_list"],
        )

    def _integrate_batches(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        method = action.method or "harmony"
        residual = self.noise.sample_qc_metric(0.05, 0.03, 0.0, 0.3)
        return IntermediateOutput(
            output_type=OutputType.EMBEDDING_SUMMARY,
            step_index=idx,
            quality_score=1.0 - residual,
            summary=f"Batch integration ({method}), residual batch effect={residual:.2f}",
            data={
                "method": method,
                "residual_batch_effect": residual,
                "n_batches": len(s.technical.batch_effects) or 1,
            },
            artifacts_available=["integrated_embedding"],
        )

    def _annotate_cell_types(
        self, action: ExperimentAction, state: FullLatentState, step_index: int
    ) -> IntermediateOutput:
        cell_populations = state.biology.cell_populations
        n_clusters = state.progress.n_clusters_found or len(cell_populations) or 1
        population_names = [p.name for p in cell_populations] if cell_populations else ["unknown"]
        annotations: Dict[str, str] = {}
        for i in range(n_clusters):
            if self.noise.coin_flip(0.15):
                name = f"unknown_{i}"
            else:
                name = str(self.noise.rng.choice(population_names))
            annotations[f"cluster_{i}"] = name
        return IntermediateOutput(
            output_type=OutputType.ANNOTATION_RESULT,
            step_index=step_index,
            data={
                "annotations": annotations,
                "method": action.method or "SingleR",
            },
            artifacts_available=["annotation_table"],
        )

    def _cluster_cells(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        n_true = len(s.biology.cell_populations) or 5
        quality = self.noise.quality_degradation(0.8, [0.95])
        n_clusters = s.last_n_clusters if s.last_n_clusters is not None else self.noise.sample_cluster_count(n_true, quality)
        cluster_names = [f"cluster_{i}" for i in range(n_clusters)]
        n_cells = s.progress.n_cells_after_filter or s.biology.n_true_cells
        sizes = self._partition_by_population(n_cells, n_clusters, s.biology.cell_populations)
        return IntermediateOutput(
            output_type=OutputType.CLUSTER_RESULT,
            step_index=idx,
            quality_score=quality,
            summary=f"Found {n_clusters} clusters",
            data={
                "n_clusters": n_clusters,
                "cluster_names": cluster_names,
                "cluster_sizes": sizes,
                "silhouette_score": self.noise.sample_qc_metric(0.35, 0.1, -1.0, 1.0),
            },
            uncertainty=abs(n_clusters - n_true) / max(n_true, 1),
            artifacts_available=["cluster_assignments", "umap_embedding"],
        )

    def _differential_expression(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        comparison = action.parameters.get("comparison", "disease_vs_healthy")
        # Fall back to the first available comparison key if the requested one
        # is absent, rather than silently returning an empty effect dict.
        if comparison not in s.biology.true_de_genes and s.biology.true_de_genes:
            comparison = next(iter(s.biology.true_de_genes))
        true_effects = s.biology.true_de_genes.get(comparison, {})

        n_cells = s.progress.n_cells_after_filter or s.biology.n_true_cells
        batch_noise = (
            sum(s.technical.batch_effects.values())
            / max(len(s.technical.batch_effects), 1)
        )
        noise_level = (
            s.technical.dropout_rate
            + 0.1 * (1.0 - s.technical.sample_quality)
            + 0.5 * batch_noise
        )
        n_subjects = s.progress.n_cohort_per_group or 4
        power_factor = min(1.0, n_subjects / 8.0)
        noise_level = noise_level / max(power_factor, 0.3)
        observed = self.noise.sample_effect_sizes(true_effects, n_cells, noise_level)

        fp_genes = self.noise.generate_false_positives(5000, 0.002 + noise_level * 0.01)
        for g in fp_genes:
            observed[g] = float(self.noise.rng.normal(0, 0.3))

        fn_genes = self.noise.generate_false_negatives(list(true_effects.keys()), 0.15)
        for g in fn_genes:
            observed.pop(g, None)

        top_genes = sorted(observed.items(), key=lambda kv: abs(kv[1]), reverse=True)[:50]
        return IntermediateOutput(
            output_type=OutputType.DE_RESULT,
            step_index=idx,
            quality_score=self.noise.quality_degradation(0.8, [1.0 - noise_level]),
            summary=f"DE analysis ({comparison}): {len(observed)} genes tested, {len(top_genes)} top hits",
            data={
                "comparison": comparison,
                "n_tested": len(observed),
                "top_genes": [
                    {"gene": g, "log2FC": round(fc, 3)} for g, fc in top_genes
                ],
                "n_significant": sum(1 for _, fc in observed.items() if abs(fc) > 0.5),
            },
            uncertainty=noise_level,
            artifacts_available=["de_table"],
        )

    def _trajectory_analysis(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        has_trajectory = s.biology.true_trajectory is not None
        quality = self.noise.quality_degradation(0.7 if has_trajectory else 0.3, [0.9])
        summary_data: Dict[str, Any] = {"method": action.method or "monocle3"}
        if has_trajectory:
            true_n_lineages = s.biology.true_trajectory.get("n_lineages", 1)
            true_branching = s.biology.true_trajectory.get("branching", False)
            # Perturb lineage count by ±1 and flip the branching flag with 20%
            # probability so the output is informative but not an exact oracle.
            noisy_n_lineages = max(1, true_n_lineages + int(self.noise.rng.choice([-1, 0, 0, 1])))
            noisy_branching = true_branching if not self.noise.coin_flip(0.20) else not true_branching
            summary_data.update({
                "n_lineages": noisy_n_lineages,
                "pseudotime_range": [0.0, 1.0],
                "branching_detected": noisy_branching,
            })
        else:
            summary_data["n_lineages"] = self.noise.sample_count(1) + 1
            summary_data["pseudotime_range"] = [0.0, 1.0]
            summary_data["branching_detected"] = self.noise.coin_flip(0.3)

        return IntermediateOutput(
            output_type=OutputType.TRAJECTORY_RESULT,
            step_index=idx,
            quality_score=quality,
            summary="Trajectory / pseudotime analysis complete",
            data=summary_data,
            uncertainty=0.2 if has_trajectory else 0.6,
            artifacts_available=["pseudotime_values", "lineage_graph"],
        )

    def _pathway_enrichment(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        true_pathways = s.biology.true_pathways
        # Pathway enrichment quality is tightly coupled to the quality of the
        # preceding DE step: more DE genes found → better gene-set coverage →
        # lower noise and fewer spurious pathway hits.
        de_genes_found = s.progress.n_de_genes_found or 0
        de_was_run = s.progress.de_performed
        if de_was_run and de_genes_found > 0:
            # Noise shrinks as the DE gene list grows (more signal in input).
            noise_level = max(0.05, 0.25 - 0.001 * min(de_genes_found, 200))
            n_fp_mean = max(1, int(5 - de_genes_found / 50))
        else:
            # Without a DE step, enrichment is unreliable.
            noise_level = 0.40
            n_fp_mean = 8

        observed: Dict[str, float] = {}
        for pw, activity in true_pathways.items():
            observed[pw] = activity + float(self.noise.rng.normal(0, noise_level))

        true_pathway_set = set(true_pathways)
        fp_candidates = [p for p in _BACKGROUND_PATHWAYS if p not in true_pathway_set]
        n_fp = self.noise.sample_count(n_fp_mean)
        if fp_candidates and n_fp > 0:
            n_sample = min(n_fp, len(fp_candidates))
            chosen = self.noise.rng.choice(
                len(fp_candidates), size=n_sample, replace=False
            )
            for idx in chosen:
                pw_name = fp_candidates[int(idx)]
                observed[pw_name] = float(self.noise.rng.uniform(0.3, 0.6))

        top = sorted(observed.items(), key=lambda kv: kv[1], reverse=True)[:15]
        top_pathway_names = [p for p, _ in top]
        recovered_true = sum(1 for p in top_pathway_names if p in true_pathway_set)
        fp_count = sum(
            1 for p in top_pathway_names
            if p in _BACKGROUND_PATHWAYS and p not in true_pathway_set
        )
        n_top = max(len(top), 1)
        fp_fraction = fp_count / n_top
        true_recall = recovered_true / max(len(true_pathway_set), 1)
        rerun_recommended = bool(
            (de_was_run and (noise_level > 0.18 or fp_fraction > 0.35))
            or (not de_was_run)
            or (true_recall < 0.25)
        )
        base_quality = 0.80 if de_was_run else 0.45
        return IntermediateOutput(
            output_type=OutputType.PATHWAY_RESULT,
            step_index=idx,
            quality_score=self.noise.quality_degradation(base_quality, [0.95]),
            summary=f"Pathway enrichment: {len(top)} significant pathways",
            data={
                "method": action.method or "GSEA",
                "top_pathways": [
                    {"pathway": p, "score": round(sc, 3)} for p, sc in top
                ],
                "quality_metrics": {
                    "de_genes_found": de_genes_found,
                    "noise_level": round(noise_level, 4),
                    "false_positive_fraction": round(fp_fraction, 4),
                    "true_pathway_recall_estimate": round(true_recall, 4),
                    "n_top_pathways": len(top),
                    "rerun_recommended": rerun_recommended,
                },
            },
            uncertainty=noise_level,
            artifacts_available=["enrichment_table"],
        )

    def _regulatory_network(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        true_net = s.biology.true_regulatory_network
        n_edges_true = sum(len(v) for v in true_net.values())
        noise_edges = self.noise.sample_count(max(5, int(n_edges_true * 0.3)))

        true_tfs = list(true_net.keys())
        # Drop ~25% of true regulators (false-negative rate).
        fn_set = set(self.noise.generate_false_negatives(true_tfs, 0.25))
        observed_tfs = [tf for tf in true_tfs if tf not in fn_set]
        # Inject realistic false-positive TFs drawn from a background pool so
        # the agent cannot distinguish true from false hits by name format.
        fp_candidates = [t for t in _NOISE_TFS if t not in set(true_tfs)]
        n_fp = self.noise.sample_count(max(2, int(len(true_tfs) * 0.5) + 2))
        if fp_candidates and n_fp > 0:
            chosen = self.noise.rng.choice(
                fp_candidates,
                size=min(n_fp, len(fp_candidates)),
                replace=False,
            )
            observed_tfs.extend(chosen.tolist())
        # Shuffle so rank order does not reveal true-vs-false identity.
        observed_tfs = self.noise.shuffle_ranking(observed_tfs, 0.5)

        return IntermediateOutput(
            output_type=OutputType.NETWORK_RESULT,
            step_index=idx,
            quality_score=self.noise.quality_degradation(0.6, [0.9]),
            summary=f"Regulatory network inferred: {n_edges_true + noise_edges} edges",
            data={
                "method": action.method or "SCENIC",
                "n_regulons": len(true_net) + self.noise.sample_count(3),
                "n_edges": n_edges_true + noise_edges,
                "top_regulators": observed_tfs[:10],
            },
            uncertainty=0.35,
            artifacts_available=["regulon_table", "grn_adjacency"],
        )

    def _marker_selection(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        true_markers = list(s.biology.true_markers)
        noise_level = 0.2
        observed_markers = [
            m for m in true_markers if not self.noise.coin_flip(noise_level)
        ]
        fp = self.noise.generate_false_positives(200, 0.01)
        observed_markers.extend(fp)
        true_marker_set = set(true_markers)
        n_candidates = len(observed_markers)
        n_true_hits = sum(1 for m in observed_markers if m in true_marker_set)
        fp_fraction = 1.0 - (n_true_hits / max(n_candidates, 1))
        rerun_recommended = bool(fp_fraction > 0.55 or n_true_hits < 2)
        return IntermediateOutput(
            output_type=OutputType.MARKER_RESULT,
            step_index=idx,
            quality_score=self.noise.quality_degradation(0.75, [0.9]),
            summary=f"Selected {len(observed_markers)} candidate markers",
            data={
                "markers": observed_markers[:20],
                "n_candidates": n_candidates,
                "quality_metrics": {
                    "estimated_true_hits": n_true_hits,
                    "estimated_false_positive_rate": round(fp_fraction, 4),
                    "noise_level": round(noise_level, 4),
                    "rerun_recommended": rerun_recommended,
                },
            },
            uncertainty=noise_level,
            artifacts_available=["marker_list"],
        )

    def _validate_marker(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        marker = action.parameters.get("marker", "unknown")
        is_true = marker in s.biology.true_markers
        validation_correct = not self.noise.coin_flip(0.1)
        validated = is_true == validation_correct
        return IntermediateOutput(
            output_type=OutputType.VALIDATION_RESULT,
            step_index=idx,
            quality_score=0.9 if validation_correct else 0.4,
            summary=f"Marker {marker}: {'validated' if validated else 'not validated'}",
            data={
                "marker": marker,
                "validated": validated,
                "assay": action.method or "qPCR",
                # Means are kept close (0.85 vs 0.45) with a wide SD (0.4)
                # so the effect size is correlated with, but not a near-perfect
                # oracle for, true marker membership.
                "effect_size": self.noise.sample_qc_metric(
                    0.85 if is_true else 0.45, 0.4, -0.5, 5.0
                ),
            },
            artifacts_available=["validation_data"],
        )

    def _design_followup(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        evidence_signals = sum([
            int(s.progress.cells_clustered),
            int(s.progress.de_performed),
            int(s.progress.trajectories_inferred),
            int(s.progress.pathways_analyzed),
            int(s.progress.networks_inferred),
            int(s.progress.markers_discovered),
            int(s.progress.markers_validated),
        ])
        return IntermediateOutput(
            output_type=OutputType.FOLLOWUP_DESIGN,
            step_index=idx,
            quality_score=min(0.75, 0.2 + 0.08 * evidence_signals),
            summary=(
                f"Follow-up experiment design proposed "
                f"(evidence_signals={evidence_signals})"
            ),
            data={
                "proposal": action.parameters,
                "evidence_signals": evidence_signals,
            },
            uncertainty=max(0.25, 0.8 - 0.08 * evidence_signals),
            artifacts_available=["followup_proposal"],
        )

    def _subagent_review(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        evidence_signals = sum([
            int(s.progress.cells_clustered),
            int(s.progress.de_performed),
            int(s.progress.trajectories_inferred),
            int(s.progress.pathways_analyzed),
            int(s.progress.networks_inferred),
            int(s.progress.markers_discovered),
            int(s.progress.markers_validated),
        ])
        return IntermediateOutput(
            output_type=OutputType.SUBAGENT_REPORT,
            step_index=idx,
            quality_score=min(0.7, 0.15 + 0.07 * evidence_signals),
            summary=f"Subagent review ({action.invoked_subagent or 'general'})",
            data={
                "subagent": action.invoked_subagent,
                "notes": "Review complete.",
                "evidence_signals": evidence_signals,
            },
            uncertainty=max(0.3, 0.85 - 0.08 * evidence_signals),
            artifacts_available=["subagent_report"],
        )

    def _synthesize_conclusion(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        return IntermediateOutput(
            output_type=OutputType.CONCLUSION,
            step_index=idx,
            summary="Conclusion synthesised from pipeline evidence",
            data={"claims": action.parameters.get("claims", [])},
            artifacts_available=["conclusion_report"],
        )

    def _assess_confounders(
        self, action: ExperimentAction, state: FullLatentState, step_index: int
    ) -> IntermediateOutput:
        """Assess confounders from biology.confounders and technical state;
        sample with small noise so some hidden ones appear in the summary."""
        confounders = state.biology.confounders or {}
        tech = state.technical
        batch_noise = (
            sum(tech.batch_effects.values()) / max(len(tech.batch_effects), 1)
            if tech.batch_effects else 0.0
        )
        noise_level = 0.05 + 0.1 * tech.dropout_rate + 0.1 * (1.0 - tech.sample_quality) + 0.15 * batch_noise
        detected: List[Dict[str, Any]] = []
        for name, strength in confounders.items():
            if self.noise.coin_flip(0.1):
                continue
            noisy_strength = float(self.noise.rng.normal(strength, noise_level))
            if noisy_strength > 0.05:
                detected.append({"name": name, "estimated_strength": round(noisy_strength, 3)})
        if self.noise.coin_flip(0.25):
            detected.append({
                "name": "batch",
                "estimated_strength": round(float(self.noise.rng.uniform(0.1, 0.4)), 3),
            })
        if self.noise.coin_flip(0.15):
            detected.append({
                "name": "capture_efficiency",
                "estimated_strength": round(float(self.noise.rng.uniform(0.05, 0.2)), 3),
            })
        quality_score = self.noise.quality_degradation(
            max(0.4, 0.9 - noise_level), [tech.sample_quality]
        )
        summary = (
            f"Confounder assessment: {len(detected)} factor(s) detected. "
            + (f"Detected: {', '.join(d['name'] for d in detected)}." if detected else "No strong confounders detected.")
        )
        recommendations = (
            "Consider batch correction and covariate stratification."
            if detected else "Confounding appears limited; proceed with caution on interpretation."
        )
        return IntermediateOutput(
            output_type=OutputType.ANALYSIS_RESULT,
            step_index=step_index,
            quality_score=quality_score,
            summary=summary,
            data={
                "detected_confounders": detected,
                "recommendations": recommendations,
                "batch_effect_level": round(batch_noise, 4),
                "dropout_rate": tech.dropout_rate,
            },
            artifacts_available=["confounder_report"],
        )

    def _stratify_by_covariate(
        self, action: ExperimentAction, state: FullLatentState, step_index: int
    ) -> IntermediateOutput:
        """Stratified analysis by the requested covariate."""
        covariate = action.parameters.get("covariate") or "batch"
        bio_strength = state.biology.confounders.get(covariate, 0.0)
        if covariate == "batch" and state.technical.batch_effects:
            batch_strength = sum(state.technical.batch_effects.values()) / len(
                state.technical.batch_effects
            )
        else:
            batch_strength = bio_strength
        combined = (bio_strength + batch_strength) / 2
        base_effect_varies = combined > 0.4
        effect_varies = (
            not base_effect_varies if self.noise.coin_flip(0.1) else base_effect_varies
        )
        base_quality = 0.5 + 0.45 * combined
        quality_score = self.noise.quality_degradation(base_quality, [0.95])
        if effect_varies:
            summary = (
                f"Stratified by {covariate}: effect varies by stratum; "
                "interpretation should account for subgroup heterogeneity."
            )
            data_extra = {"effect_heterogeneity": True, "strata_consistent": False}
        else:
            summary = (
                f"Stratified by {covariate}: results consistent across strata."
            )
            data_extra = {"effect_heterogeneity": False, "strata_consistent": True}
        return IntermediateOutput(
            output_type=OutputType.ANALYSIS_RESULT,
            step_index=step_index,
            quality_score=quality_score,
            summary=summary,
            data={
                "covariate": covariate,
                "n_strata": self.noise.sample_count(3) + 2,
                **data_extra,
            },
            artifacts_available=["stratified_results"],
        )

    def _run_sensitivity_analysis(
        self, action: ExperimentAction, state: FullLatentState, step_index: int
    ) -> IntermediateOutput:
        """Sensitivity analysis relative to a baseline step (e.g. DE)."""
        baseline_step_ref = action.parameters.get("baseline_step_ref") or "de"
        de_noise = state.last_de_noise_level or 0.2
        batches_corrected = state.progress.batches_integrated
        fragility = de_noise * (0.3 if batches_corrected else 1.0)
        p_robust = max(0.1, min(0.95, 1.0 - fragility))
        robust = self.noise.coin_flip(p_robust)
        quality_score = self.noise.quality_degradation(0.8, [0.9])
        if robust:
            summary = (
                "Sensitivity analysis: findings robust to parameter variation."
            )
            baseline_vs = {"robust": True, "n_variations_tested": self.noise.sample_count(5) + 3}
        else:
            summary = (
                "Sensitivity analysis: some conclusions sensitive to parameters; "
                "recommend reporting parameter choices."
            )
            baseline_vs = {"robust": False, "n_variations_tested": self.noise.sample_count(5) + 3}
        return IntermediateOutput(
            output_type=OutputType.ANALYSIS_RESULT,
            step_index=step_index,
            quality_score=quality_score,
            summary=summary,
            data={
                "baseline_step_ref": baseline_step_ref,
                "baseline_vs_variations": baseline_vs,
            },
            artifacts_available=["sensitivity_report"],
        )

    def _default(
        self, action: ExperimentAction, s: FullLatentState, idx: int
    ) -> IntermediateOutput:
        return IntermediateOutput(
            output_type=OutputType.FAILURE_REPORT,
            step_index=idx,
            success=False,
            summary=f"Unhandled action type: {action.action_type}",
            data={},
        )

    # ── helpers ─────────────────────────────────────────────────────────

    def _random_partition(self, total: int, k: int) -> List[int]:
        if k <= 0:
            return []
        fracs = self.noise.rng.dirichlet(alpha=[1.0] * k)
        sizes = [max(1, int(total * f)) for f in fracs]
        diff = total - sum(sizes)
        sizes[0] += diff
        return sizes

    def _partition_by_population(
        self,
        total: int,
        k: int,
        populations: list,
    ) -> List[int]:
        """Partition cells into k clusters using true population proportions
        as Dirichlet concentration parameters, so majority cell types produce
        larger clusters rather than uniformly random sizes."""
        if k <= 0:
            return []
        if populations:
            # Use true proportions as Dirichlet alpha — larger proportions
            # concentrate probability mass, yielding realistic size ratios.
            raw = [max(p.proportion, 1e-3) for p in populations]
            # Align alpha length to k: repeat/truncate as needed.
            if len(raw) >= k:
                alpha = raw[:k]
            else:
                alpha = raw + [sum(raw) / len(raw)] * (k - len(raw))
            # Scale alpha so the total magnitude is proportional to k,
            # giving reasonable Dirichlet variance.
            scale = k / max(sum(alpha), 1e-6)
            alpha = [a * scale for a in alpha]
        else:
            alpha = [1.0] * k
        fracs = self.noise.rng.dirichlet(alpha=alpha)
        sizes = [max(1, int(total * f)) for f in fracs]
        diff = total - sum(sizes)
        sizes[0] += diff
        return sizes


_HANDLERS = {
    ActionType.COLLECT_SAMPLE: OutputGenerator._collect_sample,
    ActionType.SELECT_COHORT: OutputGenerator._select_cohort,
    ActionType.PREPARE_LIBRARY: OutputGenerator._prepare_library,
    ActionType.CULTURE_CELLS: OutputGenerator._culture_cells,
    ActionType.PERTURB_GENE: OutputGenerator._perturb_gene,
    ActionType.PERTURB_COMPOUND: OutputGenerator._perturb_compound,
    ActionType.SEQUENCE_CELLS: OutputGenerator._sequence_cells,
    ActionType.RUN_QC: OutputGenerator._run_qc,
    ActionType.FILTER_DATA: OutputGenerator._filter_data,
    ActionType.NORMALIZE_DATA: OutputGenerator._normalize_data,
    ActionType.INTEGRATE_BATCHES: OutputGenerator._integrate_batches,
    ActionType.CLUSTER_CELLS: OutputGenerator._cluster_cells,
    ActionType.ANNOTATE_CELL_TYPES: OutputGenerator._annotate_cell_types,
    ActionType.DIFFERENTIAL_EXPRESSION: OutputGenerator._differential_expression,
    ActionType.TRAJECTORY_ANALYSIS: OutputGenerator._trajectory_analysis,
    ActionType.PATHWAY_ENRICHMENT: OutputGenerator._pathway_enrichment,
    ActionType.REGULATORY_NETWORK_INFERENCE: OutputGenerator._regulatory_network,
    ActionType.MARKER_SELECTION: OutputGenerator._marker_selection,
    ActionType.VALIDATE_MARKER: OutputGenerator._validate_marker,
    ActionType.DESIGN_FOLLOWUP: OutputGenerator._design_followup,
    ActionType.REQUEST_SUBAGENT_REVIEW: OutputGenerator._subagent_review,
    ActionType.SYNTHESIZE_CONCLUSION: OutputGenerator._synthesize_conclusion,
    ActionType.ASSESS_CONFOUNDERS: OutputGenerator._assess_confounders,
    ActionType.STRATIFY_BY_COVARIATE: OutputGenerator._stratify_by_covariate,
    ActionType.RUN_SENSITIVITY_ANALYSIS: OutputGenerator._run_sensitivity_analysis,
}
