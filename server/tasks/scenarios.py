"""Pre-defined biological scenarios for task generation.

Each ``Scenario`` bundles a task specification together with the matching
hidden ground-truth biology so the simulator can instantiate consistent
episodes.  The library is intentionally diverse: it covers differential
expression, trajectory inference, perturbation response, and biomarker
validation across tissues and modalities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from models import ExpectedFinding, PaperReference, TaskSpec

from server.simulator.latent_state import (
    CellPopulation,
    GeneProgram,
    LatentBiologicalState,
    TechnicalState,
)


@dataclass
class Scenario:
    """A reproducible (task, ground-truth) pair."""

    name: str
    task: TaskSpec
    biology: LatentBiologicalState
    technical: TechnicalState = field(default_factory=TechnicalState)
    hidden_failure_conditions: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    tags: List[str] = field(default_factory=list)


# ── Scenario library ────────────────────────────────────────────────────────

SCENARIO_LIBRARY: List[Scenario] = [
    # ── 1. Cardiac disease DE ───────────────────────────────────────────
    Scenario(
        name="cardiac_disease_de",
        difficulty="easy",
        tags=["de", "scRNA-seq", "cardiac"],
        task=TaskSpec(
            problem_statement=(
                "Identify differentially expressed genes between diseased "
                "and healthy cardiomyocytes using single-cell RNA sequencing."
            ),
            modality="scRNA-seq",
            organism="human",
            tissue="heart",
            conditions=["healthy", "dilated_cardiomyopathy"],
            budget_limit=80_000.0,
            time_limit_days=120.0,
            success_criteria=[
                "Identify DE genes between conditions",
                "Validate at least one candidate marker",
            ],
        ),
        biology=LatentBiologicalState(
            cell_populations=[
                CellPopulation(
                    name="cardiomyocyte",
                    proportion=0.35,
                    marker_genes=["TNNT2", "MYH7", "ACTC1"],
                    state="contractile",
                    condition_response={"dilated_cardiomyopathy": 0.8},
                ),
                CellPopulation(
                    name="fibroblast",
                    proportion=0.25,
                    marker_genes=["COL1A1", "DCN", "LUM"],
                    state="quiescent",
                    condition_response={"dilated_cardiomyopathy": 1.3},
                ),
                CellPopulation(
                    name="endothelial",
                    proportion=0.15,
                    marker_genes=["PECAM1", "VWF", "CDH5"],
                    state="quiescent",
                ),
                CellPopulation(
                    name="macrophage",
                    proportion=0.10,
                    marker_genes=["CD68", "CD163", "CSF1R"],
                    state="activated",
                    condition_response={"dilated_cardiomyopathy": 1.5},
                ),
                CellPopulation(
                    name="smooth_muscle",
                    proportion=0.15,
                    marker_genes=["ACTA2", "MYH11", "TAGLN"],
                    state="quiescent",
                ),
            ],
            true_de_genes={
                "disease_vs_healthy": {
                    "NPPA": 2.5, "NPPB": 3.1, "MYH7": 1.8,
                    "COL1A1": 1.6, "COL3A1": 1.4, "POSTN": 2.0,
                    "CCL2": 1.2, "IL6": 0.9, "TGFB1": 1.1,
                    "ANKRD1": 2.2, "XIRP2": -1.3, "MYL2": -0.8,
                },
            },
            true_pathways={
                "cardiac_muscle_contraction": 0.4,
                "extracellular_matrix_organisation": 0.85,
                "inflammatory_response": 0.7,
                "TGF_beta_signalling": 0.75,
                "apoptosis": 0.55,
            },
            true_markers=["NPPA", "NPPB", "POSTN", "COL1A1"],
            causal_mechanisms=[
                "TGF-beta-driven fibrosis",
                "inflammatory macrophage infiltration",
            ],
            confounders={"batch": 0.3},
            n_true_cells=12_000,
        ),
        technical=TechnicalState(
            batch_effects={"batch_1": 0.15, "batch_2": 0.10},
            doublet_rate=0.05,
            dropout_rate=0.08,
        ),
    ),

    # ── 2. Developmental trajectory ─────────────────────────────────────
    Scenario(
        name="hematopoiesis_trajectory",
        difficulty="medium",
        tags=["trajectory", "scRNA-seq", "hematopoiesis"],
        task=TaskSpec(
            problem_statement=(
                "Infer the developmental trajectory of hematopoietic "
                "stem cells differentiating into mature blood lineages."
            ),
            modality="scRNA-seq",
            organism="human",
            tissue="bone_marrow",
            conditions=["steady_state"],
            budget_limit=100_000.0,
            time_limit_days=150.0,
            success_criteria=[
                "Reconstruct branching lineage structure",
                "Identify key transcription factors driving fate decisions",
            ],
            paper_references=[
                PaperReference(
                    title=(
                        "Single-cell RNA-sequencing uncovers transcriptional "
                        "states and fate decisions in haematopoiesis"
                    ),
                    citation="Nature Communications (2018)",
                    doi="10.1038/s41467-017-02305-6",
                    url=(
                        "https://www.nature.com/articles/"
                        "s41467-017-02305-6"
                    ),
                ),
            ],
            expected_findings=[
                ExpectedFinding(
                    finding=(
                        "Trajectory analysis should recover branching blood "
                        "lineages rooted in HSCs."
                    ),
                    category="trajectory",
                    keywords=["HSC", "branching", "lineage", "trajectory"],
                ),
                ExpectedFinding(
                    finding=(
                        "GATA1 should appear as a driver of erythroid fate "
                        "commitment."
                    ),
                    category="regulatory_network",
                    keywords=["GATA1", "erythroid", "commitment"],
                ),
                ExpectedFinding(
                    finding=(
                        "CEBPA and SPI1 should support myeloid branch "
                        "decisions."
                    ),
                    category="regulatory_network",
                    keywords=["CEBPA", "SPI1", "myeloid", "branch"],
                ),
            ],
        ),
        biology=LatentBiologicalState(
            cell_populations=[
                CellPopulation(name="HSC", proportion=0.05,
                               marker_genes=["CD34", "KIT", "THY1"],
                               state="stem"),
                CellPopulation(name="CMP", proportion=0.10,
                               marker_genes=["CD34", "FLT3"],
                               state="progenitor"),
                CellPopulation(name="GMP", proportion=0.12,
                               marker_genes=["CSF3R", "CEBPA"],
                               state="progenitor"),
                CellPopulation(name="MEP", proportion=0.10,
                               marker_genes=["GATA1", "KLF1"],
                               state="progenitor"),
                CellPopulation(name="erythrocyte", proportion=0.20,
                               marker_genes=["HBA1", "HBB", "GYPA"],
                               state="mature"),
                CellPopulation(name="neutrophil", proportion=0.18,
                               marker_genes=["ELANE", "MPO", "CTSG"],
                               state="mature"),
                CellPopulation(name="monocyte", proportion=0.15,
                               marker_genes=["CD14", "CSF1R", "FCGR3A"],
                               state="mature"),
                CellPopulation(name="megakaryocyte", proportion=0.10,
                               marker_genes=["ITGA2B", "GP1BA"],
                               state="mature"),
            ],
            true_de_genes={},
            true_pathways={
                "hematopoietic_cell_lineage": 0.9,
                "MAPK_signalling": 0.6,
                "JAK_STAT_signalling": 0.7,
            },
            true_trajectory={
                "root": "HSC",
                "n_lineages": 3,
                "branching": True,
                "branches": [
                    ["HSC", "CMP", "GMP", "neutrophil"],
                    ["HSC", "CMP", "GMP", "monocyte"],
                    ["HSC", "MEP", "erythrocyte"],
                    ["HSC", "MEP", "megakaryocyte"],
                ],
            },
            true_regulatory_network={
                "GATA1": ["KLF1", "HBB", "HBA1", "GYPA"],
                "CEBPA": ["CSF3R", "ELANE", "MPO"],
                "SPI1": ["CSF1R", "CD14", "FCGR3A"],
                "RUNX1": ["CD34", "KIT"],
            },
            true_markers=["GATA1", "CEBPA", "SPI1"],
            causal_mechanisms=[
                "GATA1-driven erythroid commitment",
                "PU.1/CEBPA antagonism at myeloid branch point",
            ],
            confounders={},
            n_true_cells=15_000,
        ),
        technical=TechnicalState(dropout_rate=0.12, doublet_rate=0.06),
    ),

    # ── 3. Perturbation response ────────────────────────────────────────
    Scenario(
        name="perturbation_immune",
        difficulty="hard",
        tags=["perturbation", "scRNA-seq", "immune"],
        task=TaskSpec(
            problem_statement=(
                "Determine the effect of JAK inhibitor treatment on "
                "T-cell activation states in rheumatoid arthritis."
            ),
            modality="scRNA-seq",
            organism="human",
            tissue="synovial_fluid",
            conditions=["untreated_RA", "JAK_inhibitor_treated"],
            budget_limit=120_000.0,
            time_limit_days=180.0,
            prior_observations=[
                "Elevated JAK-STAT signalling observed in prior bulk RNA-seq",
            ],
            success_criteria=[
                "Quantify shift in T-cell activation states",
                "Identify pathways modulated by JAK inhibitor",
                "Propose validation strategy",
            ],
        ),
        biology=LatentBiologicalState(
            cell_populations=[
                CellPopulation(name="CD4_Th1", proportion=0.20,
                               marker_genes=["IFNG", "TBX21", "IL2"],
                               state="activated",
                               condition_response={"JAK_inhibitor_treated": 0.5}),
                CellPopulation(name="CD4_Th17", proportion=0.15,
                               marker_genes=["IL17A", "RORC", "CCR6"],
                               state="activated",
                               condition_response={"JAK_inhibitor_treated": 0.6}),
                CellPopulation(name="CD4_Treg", proportion=0.08,
                               marker_genes=["FOXP3", "IL2RA", "CTLA4"],
                               state="regulatory",
                               condition_response={"JAK_inhibitor_treated": 1.2}),
                CellPopulation(name="CD8_cytotoxic", proportion=0.18,
                               marker_genes=["GZMB", "PRF1", "CD8A"],
                               state="activated",
                               condition_response={"JAK_inhibitor_treated": 0.7}),
                CellPopulation(name="macrophage", proportion=0.15,
                               marker_genes=["CD68", "CD163", "MARCO"],
                               state="inflammatory"),
                CellPopulation(name="fibroblast", proportion=0.14,
                               marker_genes=["COL1A1", "FAP", "THY1"],
                               state="activated"),
                CellPopulation(name="B_cell", proportion=0.10,
                               marker_genes=["CD19", "MS4A1", "CD79A"],
                               state="quiescent"),
            ],
            true_de_genes={
                "treated_vs_untreated": {
                    "IFNG": -1.8, "TBX21": -1.2, "IL17A": -1.5,
                    "RORC": -0.9, "JAK1": -0.3, "STAT1": -1.0,
                    "STAT3": -0.8, "SOCS1": 1.5, "SOCS3": 1.3,
                    "FOXP3": 0.6, "IL10": 0.7,
                },
            },
            true_pathways={
                "JAK_STAT_signalling": 0.3,
                "Th1_differentiation": 0.35,
                "Th17_differentiation": 0.4,
                "cytokine_signalling": 0.45,
                "regulatory_T_cell_function": 0.7,
            },
            perturbation_effects={
                "JAK_inhibitor": {
                    "STAT1": -0.8, "STAT3": -0.7, "IFNG": -1.5,
                    "IL17A": -1.3, "SOCS1": 1.2,
                },
            },
            true_markers=["STAT1", "SOCS1", "IFNG"],
            causal_mechanisms=[
                "JAK-STAT pathway inhibition reduces Th1/Th17 activation",
                "Compensatory Treg expansion under JAK inhibition",
            ],
            confounders={"batch": 0.7, "ambient_rna": 0.5},
            n_true_cells=18_000,
        ),
        technical=TechnicalState(
            batch_effects={"batch_ctrl": 0.12, "batch_treated": 0.18},
            ambient_rna_fraction=0.07,
            dropout_rate=0.10,
        ),
        hidden_failure_conditions=[
            "High ambient RNA may confound DE in low-abundance transcripts",
        ],
    ),

    # ── 4. Biomarker validation ─────────────────────────────────────────
    Scenario(
        name="biomarker_validation_lung",
        difficulty="medium",
        tags=["biomarker", "validation", "scRNA-seq", "lung"],
        task=TaskSpec(
            problem_statement=(
                "Design a follow-up validation experiment for candidate "
                "biomarker SPP1 in idiopathic pulmonary fibrosis (IPF)."
            ),
            modality="scRNA-seq",
            organism="human",
            tissue="lung",
            conditions=["healthy", "IPF"],
            budget_limit=90_000.0,
            time_limit_days=150.0,
            prior_observations=[
                "A macrophage subpopulation shows elevated expression in IPF tissue relative to controls",
                "Pro-fibrotic macrophage enrichment has been observed in fibrotic regions by spatial profiling",
            ],
            success_criteria=[
                "Validate SPP1 as a marker for pro-fibrotic macrophages",
                "Confirm spatial localisation in fibrotic tissue",
            ],
            paper_references=[
                PaperReference(
                    title=(
                        "Proliferating SPP1/MERTK-expressing macrophages in "
                        "idiopathic pulmonary fibrosis"
                    ),
                    citation="European Respiratory Journal (2019)",
                    doi="10.1183/13993003.02441-2018",
                    pmid="31221805",
                    url="https://pubmed.ncbi.nlm.nih.gov/31221805/",
                ),
            ],
            expected_findings=[
                ExpectedFinding(
                    finding=(
                        "SPP1-positive macrophages should be enriched in IPF "
                        "fibrotic regions."
                    ),
                    category="marker",
                    keywords=["SPP1", "macrophage", "IPF", "fibrotic"],
                ),
                ExpectedFinding(
                    finding=(
                        "MERTK should co-occur with the profibrotic macrophage "
                        "state."
                    ),
                    category="marker",
                    keywords=["MERTK", "macrophage", "SPP1"],
                ),
                ExpectedFinding(
                    finding=(
                        "Extracellular matrix organization should emerge as a "
                        "top fibrotic program."
                    ),
                    category="pathway",
                    keywords=["extracellular_matrix", "fibrosis", "pathway"],
                ),
            ],
            dataset_metadata={
                "literature_grounding": "single_cell_ipf_macrophages",
            },
        ),
        biology=LatentBiologicalState(
            cell_populations=[
                CellPopulation(name="alveolar_macrophage", proportion=0.18,
                               marker_genes=["MARCO", "FABP4", "MCEMP1"],
                               state="resident"),
                CellPopulation(name="SPP1_macrophage", proportion=0.12,
                               marker_genes=["SPP1", "MERTK", "MMP9", "TREM2"],
                               state="pro-fibrotic",
                               condition_response={"IPF": 2.0}),
                CellPopulation(name="AT2", proportion=0.20,
                               marker_genes=["SFTPC", "SFTPB", "ABCA3"],
                               state="normal"),
                CellPopulation(name="fibroblast", proportion=0.22,
                               marker_genes=["COL1A1", "COL3A1", "POSTN"],
                               state="activated",
                               condition_response={"IPF": 1.5}),
                CellPopulation(name="endothelial", proportion=0.13,
                               marker_genes=["PECAM1", "CLDN5"],
                               state="quiescent"),
                CellPopulation(name="T_cell", proportion=0.15,
                               marker_genes=["CD3D", "CD3E", "IL7R"],
                               state="quiescent"),
            ],
            true_de_genes={
                "IPF_vs_healthy": {
                    "SPP1": 3.2, "MERTK": 1.4, "MMP9": 1.8, "TREM2": 1.5,
                    "COL1A1": 2.1, "COL3A1": 1.9, "POSTN": 2.4,
                    "SFTPC": -1.2, "AGER": -1.6,
                },
            },
            true_pathways={
                "extracellular_matrix_organisation": 0.9,
                "integrin_signalling": 0.75,
                "macrophage_activation": 0.8,
                "Wnt_signalling": 0.6,
            },
            true_markers=["SPP1", "MERTK", "POSTN", "MMP9"],
            causal_mechanisms=[
                "SPP1+ macrophage-driven fibroblast activation",
                "Integrin-mediated SPP1 signalling in fibrosis",
            ],
            confounders={"batch": 0.5},
            n_true_cells=14_000,
        ),
        technical=TechnicalState(
            batch_effects={"batch_1": 0.10},
            dropout_rate=0.09,
            sample_quality=0.85,
        ),
    ),

    # ── 5. Tumor microenvironment (confounders) ─────────────────────────
    Scenario(
        name="tumor_microenvironment_confounders",
        difficulty="hard",
        tags=["confounders", "tumor", "scRNA-seq"],
        task=TaskSpec(
            problem_statement=(
                "Identify tumor-associated macrophage (TAM) states and "
                "control for batch and donor confounders in tumor microenvironment."
            ),
            modality="scRNA-seq",
            organism="human",
            tissue="tumor",
            conditions=["tumor", "adjacent_normal"],
            budget_limit=110_000.0,
            time_limit_days=160.0,
            success_criteria=[
                "Identify tumor-associated macrophage states",
                "Control for batch and donor confounders",
            ],
        ),
        biology=LatentBiologicalState(
            cell_populations=[
                CellPopulation(
                    name="TAM",
                    proportion=0.22,
                    marker_genes=["CD68", "MRC1", "CD163", "IL10"],
                    state="M2_like",
                    condition_response={"tumor": 1.8},
                ),
                CellPopulation(
                    name="T_cell",
                    proportion=0.20,
                    marker_genes=["CD3D", "CD8A", "GZMB", "PDCD1"],
                    state="exhausted",
                    condition_response={"tumor": 1.2},
                ),
                CellPopulation(
                    name="fibroblast",
                    proportion=0.18,
                    marker_genes=["FAP", "ACTA2", "COL1A1", "PDPN"],
                    state="CAF",
                ),
                CellPopulation(
                    name="malignant",
                    proportion=0.25,
                    marker_genes=["EPCAM", "KRT18", "MKI67"],
                    state="tumor",
                ),
                CellPopulation(
                    name="endothelial",
                    proportion=0.15,
                    marker_genes=["PECAM1", "VWF", "CDH5"],
                    state="quiescent",
                ),
            ],
            true_de_genes={
                "tumor_vs_adjacent_normal": {
                    "MRC1": 1.5, "CD163": 1.8, "IL10": 1.2,
                    "FAP": 2.0, "COL1A1": 1.6, "EPCAM": 2.2,
                },
            },
            true_pathways={
                "immune_suppression": 0.75,
                "extracellular_matrix_organisation": 0.8,
                "angiogenesis": 0.65,
            },
            true_markers=["MRC1", "CD163", "FAP", "EPCAM"],
            causal_mechanisms=[
                "TAM-driven immune suppression",
                "CAF–tumor crosstalk",
            ],
            confounders={"batch": 0.85, "donor_age": 0.5, "composition": 0.6},
            n_true_cells=14_000,
        ),
        technical=TechnicalState(
            batch_effects={"batch_1": 0.20, "batch_2": 0.18},
            doublet_rate=0.07,
            dropout_rate=0.11,
        ),
        hidden_failure_conditions=[
            "Composition-driven differential expression can mimic true DE.",
        ],
    ),

    # ── 6. Cross-cohort biomarker ──────────────────────────────────────
    Scenario(
        name="cross_cohort_biomarker",
        difficulty="hard",
        tags=["confounders", "multicohort", "biomarker"],
        task=TaskSpec(
            problem_statement=(
                "Validate a candidate biomarker across two cohorts with "
                "different sequencing platforms."
            ),
            modality="scRNA-seq",
            organism="human",
            tissue="blood",
            conditions=["cohort_A", "cohort_B"],
            budget_limit=95_000.0,
            time_limit_days=140.0,
            success_criteria=[
                "Validate marker in both cohorts",
            ],
        ),
        biology=LatentBiologicalState(
            cell_populations=[
                CellPopulation(
                    name="population_1",
                    proportion=0.55,
                    marker_genes=["MARKER_A", "MARKER_B", "MARKER_C"],
                    state="baseline",
                ),
                CellPopulation(
                    name="population_2",
                    proportion=0.45,
                    marker_genes=["MARKER_A", "MARKER_B", "MARKER_C"],
                    state="baseline",
                ),
            ],
            true_markers=["MARKER_A", "MARKER_B", "MARKER_C"],
            causal_mechanisms=["shared_marker_expression"],
            confounders={"batch": 0.8, "platform": 0.7},
            n_true_cells=12_000,
        ),
        technical=TechnicalState(
            batch_effects={"batch_1": 0.25, "batch_2": 0.22},
            dropout_rate=0.10,
        ),
        hidden_failure_conditions=[
            "Platform and batch effects can obscure true marker signal.",
        ],
    ),

    # ── 7. Rare population under doublets ───────────────────────────────
    Scenario(
        name="rare_population_doublets",
        difficulty="hard",
        tags=["confounders", "rare_cell", "scRNA-seq"],
        task=TaskSpec(
            problem_statement=(
                "Identify and validate a rare cell population while "
                "addressing QC and confounder assessment."
            ),
            modality="scRNA-seq",
            organism="human",
            tissue="bone_marrow",
            conditions=["steady_state"],
            budget_limit=100_000.0,
            time_limit_days=150.0,
            success_criteria=[
                "Identify and validate the rare population",
                "Perform adequate QC and confounder assessment",
            ],
        ),
        biology=LatentBiologicalState(
            cell_populations=[
                CellPopulation(
                    name="rare_progenitor",
                    proportion=0.03,
                    marker_genes=["RARE1", "RARE2", "RARE3"],
                    state="rare",
                ),
                CellPopulation(
                    name="abundant_1",
                    proportion=0.35,
                    marker_genes=["AB1", "AB2", "AB3"],
                    state="abundant",
                ),
                CellPopulation(
                    name="abundant_2",
                    proportion=0.32,
                    marker_genes=["AB4", "AB5", "AB6"],
                    state="abundant",
                ),
                CellPopulation(
                    name="abundant_3",
                    proportion=0.30,
                    marker_genes=["AB7", "AB8", "AB9"],
                    state="abundant",
                ),
            ],
            true_markers=["RARE1", "RARE2", "RARE3"],
            causal_mechanisms=["rare_progenitor_fate"],
            confounders={"doublet": 0.7, "ambient_rna": 0.5},
            n_true_cells=15_000,
        ),
        technical=TechnicalState(
            doublet_rate=0.12,
            ambient_rna_fraction=0.08,
            dropout_rate=0.10,
        ),
        hidden_failure_conditions=[
            "Doublets and ambient RNA can obscure or mimic rare population signal.",
        ],
    ),
]


