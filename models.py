"""
Data models for the Bio-Experiment Planning RL Environment.

Defines the POMDP action and observation contracts for a scientific agent
that constructs biological experiment pipelines step-by-step.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation


# ── Action vocabulary ───────────────────────────────────────────────────────


class ActionType(str, Enum):
    COLLECT_SAMPLE = "collect_sample"
    SELECT_COHORT = "select_cohort"
    PREPARE_LIBRARY = "prepare_library"
    CULTURE_CELLS = "culture_cells"
    PERTURB_GENE = "perturb_gene"
    PERTURB_COMPOUND = "perturb_compound"
    SEQUENCE_CELLS = "sequence_cells"
    RUN_QC = "run_qc"
    FILTER_DATA = "filter_data"
    NORMALIZE_DATA = "normalize_data"
    INTEGRATE_BATCHES = "integrate_batches"
    CLUSTER_CELLS = "cluster_cells"
    ANNOTATE_CELL_TYPES = "annotate_cell_types"
    DIFFERENTIAL_EXPRESSION = "differential_expression"
    TRAJECTORY_ANALYSIS = "trajectory_analysis"
    PATHWAY_ENRICHMENT = "pathway_enrichment"
    REGULATORY_NETWORK_INFERENCE = "regulatory_network_inference"
    MARKER_SELECTION = "marker_selection"
    VALIDATE_MARKER = "validate_marker"
    DESIGN_FOLLOWUP = "design_followup_experiment"
    REQUEST_SUBAGENT_REVIEW = "request_subagent_review"
    SYNTHESIZE_CONCLUSION = "synthesize_conclusion"
    ASSESS_CONFOUNDERS = "assess_confounders"
    STRATIFY_BY_COVARIATE = "stratify_by_covariate"
    RUN_SENSITIVITY_ANALYSIS = "run_sensitivity_analysis"


WET_LAB_ACTIONS = frozenset({
    ActionType.COLLECT_SAMPLE,
    ActionType.SELECT_COHORT,
    ActionType.PREPARE_LIBRARY,
    ActionType.CULTURE_CELLS,
    ActionType.PERTURB_GENE,
    ActionType.PERTURB_COMPOUND,
    ActionType.SEQUENCE_CELLS,
    ActionType.VALIDATE_MARKER,
})

COMPUTATIONAL_ACTIONS = frozenset({
    ActionType.RUN_QC,
    ActionType.FILTER_DATA,
    ActionType.NORMALIZE_DATA,
    ActionType.INTEGRATE_BATCHES,
    ActionType.CLUSTER_CELLS,
    ActionType.ANNOTATE_CELL_TYPES,
    ActionType.DIFFERENTIAL_EXPRESSION,
    ActionType.TRAJECTORY_ANALYSIS,
    ActionType.PATHWAY_ENRICHMENT,
    ActionType.REGULATORY_NETWORK_INFERENCE,
    ActionType.MARKER_SELECTION,
    ActionType.ASSESS_CONFOUNDERS,
    ActionType.STRATIFY_BY_COVARIATE,
    ActionType.RUN_SENSITIVITY_ANALYSIS,
})

INFERENCE_ACTIONS = frozenset({
    ActionType.ASSESS_CONFOUNDERS,
    ActionType.STRATIFY_BY_COVARIATE,
    ActionType.RUN_SENSITIVITY_ANALYSIS,
})

META_ACTIONS = frozenset({
    ActionType.DESIGN_FOLLOWUP,
    ActionType.REQUEST_SUBAGENT_REVIEW,
    ActionType.SYNTHESIZE_CONCLUSION,
})


# ── Tool, Assay & Modality Registries ──────────────────────────────────────


class ToolCategory(str, Enum):
    ALIGNMENT = "alignment"
    PREPROCESSING = "preprocessing"
    NORMALIZATION = "normalization"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    CLUSTERING = "clustering"
    DIFFERENTIAL_EXPRESSION = "differential_expression"
    TRAJECTORY = "trajectory"
    GENE_REGULATORY_NETWORK = "gene_regulatory_network"
    CELL_COMMUNICATION = "cell_communication"
    SPATIAL = "spatial"
    MULTIMODAL_INTEGRATION = "multimodal_integration"
    GENE_SET_ANALYSIS = "gene_set_analysis"
    VARIANT_CALLING = "variant_calling"
    PEAK_CALLING = "peak_calling"
    IMPUTATION = "imputation"
    BATCH_CORRECTION = "batch_correction"
    CELL_TYPE_ANNOTATION = "cell_type_annotation"
    SIMULATION = "simulation"
    VISUALIZATION = "visualization"
    QUALITY_CONTROL = "quality_control"
    PERTURBATION_ANALYSIS = "perturbation_analysis"
    STATISTICAL_INFERENCE = "statistical_inference"
    KNOWLEDGE_BASE = "knowledge_base"


class ToolSpec(BaseModel):
    """Registry entry describing a bioinformatics tool."""

    name: str
    category: ToolCategory
    modalities: List[str] = Field(default_factory=list)
    description: str = ""
    input_types: List[str] = Field(default_factory=list)
    output_types: List[str] = Field(default_factory=list)
    typical_runtime_hours: float = 0.1
    typical_cost_usd: float = 0.0
    requires_gpu: bool = False
    open_source: bool = True


TOOL_REGISTRY: Dict[str, ToolSpec] = {
    # ── Alignment & quantification ──
    "CellRanger": ToolSpec(
        name="CellRanger",
        category=ToolCategory.ALIGNMENT,
        modalities=["scRNA-seq", "scATAC-seq", "CITE-seq", "scMultiome"],
        description="10x Genomics pipeline for alignment, barcode demux, and counting",
        input_types=["fastq"],
        output_types=["count_matrix", "bam"],
        typical_runtime_hours=4.0,
        open_source=False,
    ),
    "STARsolo": ToolSpec(
        name="STARsolo",
        category=ToolCategory.ALIGNMENT,
        modalities=["scRNA-seq", "scATAC-seq"],
        description="Drop-seq / 10x-compatible aligner built into STAR",
        input_types=["fastq"],
        output_types=["count_matrix", "bam"],
        typical_runtime_hours=3.0,
    ),
    "kallisto_bustools": ToolSpec(
        name="kallisto_bustools",
        category=ToolCategory.ALIGNMENT,
        modalities=["scRNA-seq"],
        description="Pseudoalignment-based lightweight quantification",
        input_types=["fastq"],
        output_types=["count_matrix"],
        typical_runtime_hours=1.0,
    ),
    "Salmon_alevin": ToolSpec(
        name="Salmon_alevin",
        category=ToolCategory.ALIGNMENT,
        modalities=["scRNA-seq"],
        description="Quasi-mapping quantification for single-cell RNA-seq",
        input_types=["fastq"],
        output_types=["count_matrix"],
        typical_runtime_hours=1.5,
    ),
    "spaceranger": ToolSpec(
        name="spaceranger",
        category=ToolCategory.ALIGNMENT,
        modalities=["spatial_transcriptomics"],
        description="10x Visium spatial alignment and quantification",
        input_types=["fastq", "image"],
        output_types=["count_matrix", "spatial_coords"],
        typical_runtime_hours=3.0,
        open_source=False,
    ),
    # ── Preprocessing / analysis frameworks ──
    "Scanpy": ToolSpec(
        name="Scanpy",
        category=ToolCategory.PREPROCESSING,
        modalities=["scRNA-seq", "scATAC-seq", "spatial_transcriptomics"],
        description="Python single-cell analysis framework",
        input_types=["count_matrix", "h5ad"],
        output_types=["h5ad", "embedding", "cluster_result"],
        typical_runtime_hours=0.5,
    ),
    "Seurat": ToolSpec(
        name="Seurat",
        category=ToolCategory.PREPROCESSING,
        modalities=["scRNA-seq", "CITE-seq", "spatial_transcriptomics", "scATAC-seq"],
        description="R single-cell analysis toolkit with multimodal support",
        input_types=["count_matrix", "h5seurat"],
        output_types=["h5seurat", "embedding", "cluster_result"],
        typical_runtime_hours=0.5,
    ),
    "Bioconductor_SingleCellExperiment": ToolSpec(
        name="Bioconductor_SingleCellExperiment",
        category=ToolCategory.PREPROCESSING,
        modalities=["scRNA-seq"],
        description="R/Bioconductor framework for single-cell experiments",
        input_types=["count_matrix"],
        output_types=["sce_object"],
        typical_runtime_hours=0.3,
    ),
    # ── Normalization ──
    "scran": ToolSpec(
        name="scran",
        category=ToolCategory.NORMALIZATION,
        modalities=["scRNA-seq"],
        description="Pool-based size-factor normalization",
        input_types=["count_matrix"],
        output_types=["normalized_matrix"],
    ),
    "sctransform": ToolSpec(
        name="sctransform",
        category=ToolCategory.NORMALIZATION,
        modalities=["scRNA-seq"],
        description="Variance-stabilizing transformation via regularized NB regression",
        input_types=["count_matrix"],
        output_types=["normalized_matrix"],
    ),
    # ── Dimensionality reduction ──
    "scVI": ToolSpec(
        name="scVI",
        category=ToolCategory.DIMENSIONALITY_REDUCTION,
        modalities=["scRNA-seq", "CITE-seq", "scATAC-seq"],
        description="Deep generative model for scRNA-seq (variational inference)",
        input_types=["count_matrix"],
        output_types=["latent_embedding"],
        requires_gpu=True,
    ),
    "UMAP": ToolSpec(
        name="UMAP",
        category=ToolCategory.DIMENSIONALITY_REDUCTION,
        modalities=["scRNA-seq", "scATAC-seq", "CITE-seq", "spatial_transcriptomics"],
        description="Uniform manifold approximation for 2D/3D visualization",
        input_types=["pca_embedding", "latent_embedding"],
        output_types=["2d_embedding"],
    ),
    # ── Clustering ──
    "Leiden": ToolSpec(
        name="Leiden",
        category=ToolCategory.CLUSTERING,
        modalities=["scRNA-seq", "scATAC-seq", "CITE-seq"],
        description="Community detection via the Leiden algorithm",
        input_types=["knn_graph"],
        output_types=["cluster_result"],
    ),
    "Louvain": ToolSpec(
        name="Louvain",
        category=ToolCategory.CLUSTERING,
        modalities=["scRNA-seq", "scATAC-seq"],
        description="Community detection via Louvain modularity optimization",
        input_types=["knn_graph"],
        output_types=["cluster_result"],
    ),
    # ── Differential expression ──
    "DESeq2": ToolSpec(
        name="DESeq2",
        category=ToolCategory.DIFFERENTIAL_EXPRESSION,
        modalities=["bulk_rna_seq", "scRNA-seq"],
        description="Negative binomial GLM-based differential expression",
        input_types=["count_matrix"],
        output_types=["de_result"],
    ),
    "MAST": ToolSpec(
        name="MAST",
        category=ToolCategory.DIFFERENTIAL_EXPRESSION,
        modalities=["scRNA-seq"],
        description="Two-part hurdle model for scRNA-seq DE testing",
        input_types=["count_matrix"],
        output_types=["de_result"],
    ),
    "edgeR": ToolSpec(
        name="edgeR",
        category=ToolCategory.DIFFERENTIAL_EXPRESSION,
        modalities=["bulk_rna_seq", "scRNA-seq"],
        description="Empirical Bayes quasi-likelihood DE testing",
        input_types=["count_matrix"],
        output_types=["de_result"],
    ),
    "Wilcoxon": ToolSpec(
        name="Wilcoxon",
        category=ToolCategory.DIFFERENTIAL_EXPRESSION,
        modalities=["scRNA-seq"],
        description="Rank-sum test for marker gene detection",
        input_types=["count_matrix"],
        output_types=["de_result"],
    ),
    # ── Trajectory & RNA velocity ──
    "Monocle3": ToolSpec(
        name="Monocle3",
        category=ToolCategory.TRAJECTORY,
        modalities=["scRNA-seq"],
        description="Reversed graph embedding for pseudotime trajectories",
        input_types=["count_matrix", "embedding"],
        output_types=["trajectory_result", "pseudotime"],
    ),
    "scVelo": ToolSpec(
        name="scVelo",
        category=ToolCategory.TRAJECTORY,
        modalities=["scRNA-seq"],
        description="RNA velocity estimation via spliced/unspliced dynamics",
        input_types=["count_matrix"],
        output_types=["velocity_result"],
    ),
    "CellRank": ToolSpec(
        name="CellRank",
        category=ToolCategory.TRAJECTORY,
        modalities=["scRNA-seq"],
        description="Fate probability estimation combining velocity and transcriptomics",
        input_types=["velocity_result", "count_matrix"],
        output_types=["fate_probabilities"],
    ),
    "Slingshot": ToolSpec(
        name="Slingshot",
        category=ToolCategory.TRAJECTORY,
        modalities=["scRNA-seq"],
        description="Minimum spanning tree-based trajectory inference",
        input_types=["embedding", "cluster_result"],
        output_types=["trajectory_result", "pseudotime"],
    ),
    "PAGA": ToolSpec(
        name="PAGA",
        category=ToolCategory.TRAJECTORY,
        modalities=["scRNA-seq"],
        description="Partition-based graph abstraction for topology estimation",
        input_types=["knn_graph", "cluster_result"],
        output_types=["trajectory_result"],
    ),
    # ── Gene regulatory networks ──
    "SCENIC": ToolSpec(
        name="SCENIC",
        category=ToolCategory.GENE_REGULATORY_NETWORK,
        modalities=["scRNA-seq"],
        description="Single-cell regulatory network inference and clustering",
        input_types=["count_matrix"],
        output_types=["regulon_result", "network_result"],
        typical_runtime_hours=6.0,
    ),
    "CellOracle": ToolSpec(
        name="CellOracle",
        category=ToolCategory.GENE_REGULATORY_NETWORK,
        modalities=["scRNA-seq", "scATAC-seq", "scMultiome"],
        description="GRN-based in-silico perturbation prediction",
        input_types=["count_matrix", "peak_matrix"],
        output_types=["network_result", "perturbation_prediction"],
        typical_runtime_hours=4.0,
    ),
    # ── Cell-cell communication ──
    "CellChat": ToolSpec(
        name="CellChat",
        category=ToolCategory.CELL_COMMUNICATION,
        modalities=["scRNA-seq", "spatial_transcriptomics"],
        description="Ligand-receptor interaction inference with communication patterns",
        input_types=["count_matrix", "cluster_result"],
        output_types=["communication_result"],
    ),
    "NicheNet": ToolSpec(
        name="NicheNet",
        category=ToolCategory.CELL_COMMUNICATION,
        modalities=["scRNA-seq"],
        description="Ligand-target link prediction using prior knowledge",
        input_types=["count_matrix", "de_result"],
        output_types=["communication_result"],
    ),
    "LIANA": ToolSpec(
        name="LIANA",
        category=ToolCategory.CELL_COMMUNICATION,
        modalities=["scRNA-seq", "spatial_transcriptomics"],
        description="Framework unifying multiple ligand-receptor methods",
        input_types=["count_matrix", "cluster_result"],
        output_types=["communication_result"],
    ),
    # ── Spatial analysis ──
    "squidpy": ToolSpec(
        name="squidpy",
        category=ToolCategory.SPATIAL,
        modalities=["spatial_transcriptomics"],
        description="Spatial omics analysis (neighborhood, co-occurrence, image features)",
        input_types=["count_matrix", "spatial_coords"],
        output_types=["spatial_result"],
    ),
    "cell2location": ToolSpec(
        name="cell2location",
        category=ToolCategory.SPATIAL,
        modalities=["spatial_transcriptomics"],
        description="Spatial deconvolution mapping cell types to tissue locations",
        input_types=["count_matrix", "spatial_coords", "reference_h5ad"],
        output_types=["deconvolution_result"],
        requires_gpu=True,
    ),
    "BANKSY": ToolSpec(
        name="BANKSY",
        category=ToolCategory.SPATIAL,
        modalities=["spatial_transcriptomics"],
        description="Spatially-aware clustering combining cell and neighbor features",
        input_types=["count_matrix", "spatial_coords"],
        output_types=["cluster_result"],
    ),
    # ── Multimodal integration ──
    "Harmony": ToolSpec(
        name="Harmony",
        category=ToolCategory.BATCH_CORRECTION,
        modalities=["scRNA-seq", "scATAC-seq", "CITE-seq"],
        description="Fast iterative batch correction on PCA embeddings",
        input_types=["pca_embedding"],
        output_types=["corrected_embedding"],
    ),
    "scanorama": ToolSpec(
        name="scanorama",
        category=ToolCategory.BATCH_CORRECTION,
        modalities=["scRNA-seq"],
        description="Panoramic stitching of scRNA-seq batches",
        input_types=["count_matrix"],
        output_types=["corrected_embedding", "corrected_matrix"],
    ),
    "BBKNN": ToolSpec(
        name="BBKNN",
        category=ToolCategory.BATCH_CORRECTION,
        modalities=["scRNA-seq"],
        description="Batch-balanced KNN graph construction",
        input_types=["pca_embedding"],
        output_types=["knn_graph"],
    ),
    "WNN": ToolSpec(
        name="WNN",
        category=ToolCategory.MULTIMODAL_INTEGRATION,
        modalities=["CITE-seq", "scMultiome"],
        description="Weighted nearest neighbors for multimodal integration (Seurat v4+)",
        input_types=["rna_embedding", "protein_embedding"],
        output_types=["multimodal_embedding"],
    ),
    "MOFA+": ToolSpec(
        name="MOFA+",
        category=ToolCategory.MULTIMODAL_INTEGRATION,
        modalities=["scMultiome", "CITE-seq"],
        description="Multi-omics factor analysis for unsupervised integration",
        input_types=["count_matrix", "peak_matrix"],
        output_types=["factor_result"],
    ),
    "ArchR": ToolSpec(
        name="ArchR",
        category=ToolCategory.PREPROCESSING,
        modalities=["scATAC-seq", "scMultiome"],
        description="Full-featured scATAC-seq analysis framework in R",
        input_types=["fragments", "bam"],
        output_types=["peak_matrix", "gene_activity_matrix"],
        typical_runtime_hours=2.0,
    ),
    "Signac": ToolSpec(
        name="Signac",
        category=ToolCategory.PREPROCESSING,
        modalities=["scATAC-seq", "scMultiome"],
        description="Seurat extension for chromatin accessibility analysis",
        input_types=["fragments", "peak_matrix"],
        output_types=["peak_matrix", "motif_result"],
    ),
    "chromVAR": ToolSpec(
        name="chromVAR",
        category=ToolCategory.PEAK_CALLING,
        modalities=["scATAC-seq", "scMultiome"],
        description="TF motif accessibility deviation scoring",
        input_types=["peak_matrix"],
        output_types=["motif_deviation_scores"],
    ),
    # ── Gene set / pathway analysis ──
    "GSEA": ToolSpec(
        name="GSEA",
        category=ToolCategory.GENE_SET_ANALYSIS,
        modalities=["bulk_rna_seq", "scRNA-seq"],
        description="Gene Set Enrichment Analysis (preranked or phenotype-based)",
        input_types=["de_result", "ranked_gene_list"],
        output_types=["pathway_result"],
    ),
    "clusterProfiler": ToolSpec(
        name="clusterProfiler",
        category=ToolCategory.GENE_SET_ANALYSIS,
        modalities=["bulk_rna_seq", "scRNA-seq"],
        description="ORA & GSEA with GO, KEGG, Reactome, and custom gene sets",
        input_types=["de_result", "gene_list"],
        output_types=["pathway_result"],
    ),
    "decoupleR": ToolSpec(
        name="decoupleR",
        category=ToolCategory.GENE_SET_ANALYSIS,
        modalities=["scRNA-seq", "bulk_rna_seq", "spatial_transcriptomics"],
        description="Unified framework for functional activity inference (TF, pathway)",
        input_types=["count_matrix", "de_result"],
        output_types=["activity_scores"],
    ),
    # ── Cell type annotation ──
    "celltypist": ToolSpec(
        name="celltypist",
        category=ToolCategory.CELL_TYPE_ANNOTATION,
        modalities=["scRNA-seq"],
        description="Automated cell type classification with pre-trained models",
        input_types=["count_matrix"],
        output_types=["annotation_result"],
    ),
    "SingleR": ToolSpec(
        name="SingleR",
        category=ToolCategory.CELL_TYPE_ANNOTATION,
        modalities=["scRNA-seq"],
        description="Reference-based cell type annotation using correlation",
        input_types=["count_matrix", "reference_dataset"],
        output_types=["annotation_result"],
    ),
    "scArches": ToolSpec(
        name="scArches",
        category=ToolCategory.CELL_TYPE_ANNOTATION,
        modalities=["scRNA-seq", "scATAC-seq", "CITE-seq"],
        description="Reference mapping and label transfer via deep learning",
        input_types=["count_matrix", "reference_model"],
        output_types=["annotation_result", "latent_embedding"],
        requires_gpu=True,
    ),
    # ── Imputation ──
    "MAGIC": ToolSpec(
        name="MAGIC",
        category=ToolCategory.IMPUTATION,
        modalities=["scRNA-seq"],
        description="Markov affinity-based graph imputation of dropout zeros",
        input_types=["count_matrix"],
        output_types=["imputed_matrix"],
    ),
    # ── Perturbation analysis ──
    "MILO": ToolSpec(
        name="MILO",
        category=ToolCategory.PERTURBATION_ANALYSIS,
        modalities=["scRNA-seq"],
        description="Differential abundance testing on KNN graph neighborhoods",
        input_types=["count_matrix", "knn_graph"],
        output_types=["da_result"],
    ),
    "Mixscape": ToolSpec(
        name="Mixscape",
        category=ToolCategory.PERTURBATION_ANALYSIS,
        modalities=["Perturb-seq", "CROP-seq"],
        description="Seurat extension for CRISPR screen perturbation analysis",
        input_types=["count_matrix", "guide_assignments"],
        output_types=["perturbation_result"],
    ),
    "MIMOSCA": ToolSpec(
        name="MIMOSCA",
        category=ToolCategory.PERTURBATION_ANALYSIS,
        modalities=["Perturb-seq", "CROP-seq"],
        description="Multi-input multi-output single-cell analysis for screens",
        input_types=["count_matrix", "guide_assignments"],
        output_types=["perturbation_result"],
    ),
    # ── Quality control ──
    "scrublet": ToolSpec(
        name="scrublet",
        category=ToolCategory.QUALITY_CONTROL,
        modalities=["scRNA-seq"],
        description="Computational doublet detection via synthetic doublets",
        input_types=["count_matrix"],
        output_types=["doublet_scores"],
    ),
    "DoubletFinder": ToolSpec(
        name="DoubletFinder",
        category=ToolCategory.QUALITY_CONTROL,
        modalities=["scRNA-seq"],
        description="Artificial nearest-neighbor doublet detection",
        input_types=["count_matrix"],
        output_types=["doublet_scores"],
    ),
    "SoupX": ToolSpec(
        name="SoupX",
        category=ToolCategory.QUALITY_CONTROL,
        modalities=["scRNA-seq"],
        description="Ambient RNA contamination estimation and removal",
        input_types=["count_matrix", "raw_count_matrix"],
        output_types=["corrected_matrix"],
    ),
    "DecontX": ToolSpec(
        name="DecontX",
        category=ToolCategory.QUALITY_CONTROL,
        modalities=["scRNA-seq"],
        description="Bayesian ambient RNA decontamination",
        input_types=["count_matrix"],
        output_types=["corrected_matrix"],
    ),
    # ── Simulation ──
    "Splatter": ToolSpec(
        name="Splatter",
        category=ToolCategory.SIMULATION,
        modalities=["scRNA-seq"],
        description="Flexible scRNA-seq data simulation framework",
        input_types=["simulation_params"],
        output_types=["simulated_count_matrix"],
    ),
    # ── Variant calling ──
    "GATK": ToolSpec(
        name="GATK",
        category=ToolCategory.VARIANT_CALLING,
        modalities=["genomics"],
        description="Genome Analysis Toolkit for variant discovery and genotyping",
        input_types=["bam", "reference_fasta"],
        output_types=["vcf", "gvcf"],
        typical_runtime_hours=8.0,
    ),
    "bcftools": ToolSpec(
        name="bcftools",
        category=ToolCategory.VARIANT_CALLING,
        modalities=["genomics"],
        description="Variant calling and manipulation for VCF/BCF files",
        input_types=["bam", "vcf"],
        output_types=["vcf", "bcf"],
        typical_runtime_hours=2.0,
    ),
    # ── Knowledge base ──
    "Ensembl": ToolSpec(
        name="Ensembl",
        category=ToolCategory.KNOWLEDGE_BASE,
        modalities=["bulk_rna_seq", "scRNA-seq", "genomics"],
        description="Reference genome and annotation lookup",
        input_types=["gene_id", "transcript_id"],
        output_types=["annotation", "sequence"],
        typical_cost_usd=0,
    ),
    "MSigDB": ToolSpec(
        name="MSigDB",
        category=ToolCategory.KNOWLEDGE_BASE,
        modalities=["bulk_rna_seq", "scRNA-seq"],
        description="Molecular signatures and gene set collections for enrichment",
        input_types=["gene_list"],
        output_types=["gene_set_metadata"],
        typical_cost_usd=0,
    ),
    # ── Statistical inference ──
    "mixed_models": ToolSpec(
        name="mixed_models",
        category=ToolCategory.STATISTICAL_INFERENCE,
        modalities=["bulk_rna_seq", "scRNA-seq"],
        description="Mixed-effects models for confounder adjustment and repeated measures",
        input_types=["count_matrix", "metadata"],
        output_types=["model_result", "adjusted_estimates"],
        typical_runtime_hours=0.5,
    ),
}


class Modality(str, Enum):
    SCRNA_SEQ = "scRNA-seq"
    SCATAC_SEQ = "scATAC-seq"
    CITE_SEQ = "CITE-seq"
    SPATIAL_TRANSCRIPTOMICS = "spatial_transcriptomics"
    BULK_RNA_SEQ = "bulk_rna_seq"
    SCRNA_MULTIOME = "scMultiome"
    PERTURB_SEQ = "Perturb-seq"
    CROP_SEQ = "CROP-seq"
    SMART_SEQ2 = "Smart-seq2"
    SLIDE_SEQ = "Slide-seq"
    MERFISH = "MERFISH"
    SEQFISH = "seqFISH"
    PATCH_SEQ = "Patch-seq"
    SHARE_SEQ = "SHARE-seq"
    SNARE_SEQ = "SNARE-seq"
    SC_HI_C = "scHi-C"
    SCBS_SEQ = "scBS-seq"
    SCNMT_SEQ = "scNMT-seq"


class ModalitySpec(BaseModel):
    """Registry entry for a single-cell or bulk assay modality."""

    name: str
    modality: Modality
    measurement: str = ""
    resolution: str = "single-cell"
    multiplexable: bool = False
    typical_cells: str = "1k-20k"
    typical_cost_per_sample_usd: float = 5000.0
    compatible_tools: List[str] = Field(default_factory=list)
    description: str = ""


MODALITY_REGISTRY: Dict[str, ModalitySpec] = {
    "scRNA-seq": ModalitySpec(
        name="scRNA-seq",
        modality=Modality.SCRNA_SEQ,
        measurement="mRNA transcripts",
        typical_cells="5k-20k",
        typical_cost_per_sample_usd=5000.0,
        compatible_tools=[
            "CellRanger", "STARsolo", "kallisto_bustools", "Scanpy", "Seurat",
            "scVI", "Leiden", "DESeq2", "MAST", "Monocle3", "scVelo", "SCENIC",
            "CellChat", "GSEA", "celltypist", "scrublet",
        ],
        description="Droplet-based single-cell RNA sequencing (e.g. 10x Chromium)",
    ),
    "scATAC-seq": ModalitySpec(
        name="scATAC-seq",
        modality=Modality.SCATAC_SEQ,
        measurement="open chromatin regions",
        typical_cells="5k-15k",
        typical_cost_per_sample_usd=6000.0,
        compatible_tools=[
            "CellRanger", "ArchR", "Signac", "chromVAR", "Scanpy", "Leiden",
        ],
        description="Single-cell Assay for Transposase-Accessible Chromatin",
    ),
    "CITE-seq": ModalitySpec(
        name="CITE-seq",
        modality=Modality.CITE_SEQ,
        measurement="mRNA + surface proteins (ADT)",
        multiplexable=True,
        typical_cells="5k-20k",
        typical_cost_per_sample_usd=8000.0,
        compatible_tools=[
            "CellRanger", "Seurat", "WNN", "MOFA+", "Scanpy", "Leiden",
        ],
        description="Cellular Indexing of Transcriptomes and Epitopes by Sequencing",
    ),
    "spatial_transcriptomics": ModalitySpec(
        name="spatial_transcriptomics",
        modality=Modality.SPATIAL_TRANSCRIPTOMICS,
        measurement="spatially resolved transcripts",
        resolution="spot (55µm) or subcellular",
        typical_cells="1k-10k spots",
        typical_cost_per_sample_usd=7000.0,
        compatible_tools=[
            "spaceranger", "squidpy", "cell2location", "BANKSY", "Scanpy", "Seurat",
        ],
        description="Spatially resolved transcriptomics (Visium, MERFISH, Slide-seq, etc.)",
    ),
    "bulk_rna_seq": ModalitySpec(
        name="bulk_rna_seq",
        modality=Modality.BULK_RNA_SEQ,
        measurement="aggregate mRNA across cells",
        resolution="bulk",
        typical_cells="N/A",
        typical_cost_per_sample_usd=500.0,
        compatible_tools=["DESeq2", "edgeR", "GSEA", "clusterProfiler", "mixed_models", "MSigDB", "Ensembl"],
        description="Standard bulk RNA sequencing",
    ),
    "scMultiome": ModalitySpec(
        name="scMultiome",
        modality=Modality.SCRNA_MULTIOME,
        measurement="mRNA + open chromatin (joint)",
        typical_cells="5k-15k",
        typical_cost_per_sample_usd=10000.0,
        compatible_tools=[
            "CellRanger", "ArchR", "Signac", "Seurat", "MOFA+", "CellOracle",
        ],
        description="10x Multiome (joint scRNA + scATAC from same cell)",
    ),
    "Perturb-seq": ModalitySpec(
        name="Perturb-seq",
        modality=Modality.PERTURB_SEQ,
        measurement="mRNA + CRISPR guide assignment",
        multiplexable=True,
        typical_cells="10k-100k",
        typical_cost_per_sample_usd=15000.0,
        compatible_tools=[
            "CellRanger", "Scanpy", "Seurat", "Mixscape", "MIMOSCA",
        ],
        description="Pooled CRISPR screens with single-cell RNA readout",
    ),
    "CROP-seq": ModalitySpec(
        name="CROP-seq",
        modality=Modality.CROP_SEQ,
        measurement="mRNA + CRISPR guide assignment",
        multiplexable=True,
        typical_cells="10k-50k",
        typical_cost_per_sample_usd=12000.0,
        compatible_tools=[
            "CellRanger", "Scanpy", "Seurat", "Mixscape", "MIMOSCA",
        ],
        description="CRISPR dropout screen with single-cell RNA readout",
    ),
    "Smart-seq2": ModalitySpec(
        name="Smart-seq2",
        modality=Modality.SMART_SEQ2,
        measurement="full-length mRNA transcripts",
        typical_cells="100-1000",
        typical_cost_per_sample_usd=10000.0,
        compatible_tools=["Scanpy", "Seurat", "DESeq2", "MAST", "Monocle3"],
        description="Plate-based full-length scRNA-seq with high sensitivity",
    ),
    "MERFISH": ModalitySpec(
        name="MERFISH",
        modality=Modality.MERFISH,
        measurement="in situ mRNA (imaging-based)",
        resolution="subcellular",
        typical_cells="10k-1M",
        typical_cost_per_sample_usd=20000.0,
        compatible_tools=["squidpy", "Scanpy", "BANKSY"],
        description="Multiplexed Error-Robust FISH for spatial transcriptomics",
    ),
    "Slide-seq": ModalitySpec(
        name="Slide-seq",
        modality=Modality.SLIDE_SEQ,
        measurement="spatially resolved mRNA (bead array)",
        resolution="10µm",
        typical_cells="10k-50k beads",
        typical_cost_per_sample_usd=8000.0,
        compatible_tools=["squidpy", "cell2location", "Scanpy"],
        description="Near-cellular spatial transcriptomics on bead arrays",
    ),
    "Patch-seq": ModalitySpec(
        name="Patch-seq",
        modality=Modality.PATCH_SEQ,
        measurement="mRNA + electrophysiology + morphology",
        typical_cells="10-500",
        typical_cost_per_sample_usd=50000.0,
        compatible_tools=["Scanpy", "Seurat"],
        description="Combined patch-clamp electrophysiology and scRNA-seq",
    ),
    "scHi-C": ModalitySpec(
        name="scHi-C",
        modality=Modality.SC_HI_C,
        measurement="3D chromatin contacts",
        typical_cells="1k-10k",
        typical_cost_per_sample_usd=15000.0,
        compatible_tools=["Scanpy"],
        description="Single-cell chromosome conformation capture",
    ),
    "scBS-seq": ModalitySpec(
        name="scBS-seq",
        modality=Modality.SCBS_SEQ,
        measurement="DNA methylation (CpG)",
        typical_cells="100-5k",
        typical_cost_per_sample_usd=12000.0,
        compatible_tools=["Scanpy"],
        description="Single-cell bisulfite sequencing for DNA methylation",
    ),
    "scNMT-seq": ModalitySpec(
        name="scNMT-seq",
        modality=Modality.SCNMT_SEQ,
        measurement="nucleosome + methylation + transcription (joint)",
        typical_cells="100-1k",
        typical_cost_per_sample_usd=25000.0,
        compatible_tools=["MOFA+", "Scanpy"],
        description="Joint single-cell nucleosome, methylation, and transcription",
    ),
}


class AssayCategory(str, Enum):
    SEQUENCING = "sequencing"
    IMAGING = "imaging"
    PERTURBATION = "perturbation"
    FUNCTIONAL = "functional"
    EPIGENOMICS = "epigenomics"
    PROTEOMICS = "proteomics"
    METABOLOMICS = "metabolomics"


class AssaySpec(BaseModel):
    """Registry entry for a laboratory assay or protocol."""

    name: str
    category: AssayCategory
    modalities: List[str] = Field(default_factory=list)
    description: str = ""
    typical_duration_days: float = 1.0
    typical_cost_usd: float = 1000.0
    requires_live_cells: bool = False
    requires_fresh_tissue: bool = False
    throughput: str = "medium"
    outputs: List[str] = Field(default_factory=list)


ASSAY_REGISTRY: Dict[str, AssaySpec] = {
    "10x_chromium": AssaySpec(
        name="10x_chromium",
        category=AssayCategory.SEQUENCING,
        modalities=["scRNA-seq", "scATAC-seq", "CITE-seq", "scMultiome"],
        description="10x Genomics Chromium droplet-based single-cell partitioning",
        typical_duration_days=2.0,
        typical_cost_usd=5000.0,
        requires_live_cells=True,
        throughput="high (500-20k cells)",
        outputs=["fastq", "count_matrix"],
    ),
    "smart-seq2": AssaySpec(
        name="smart-seq2",
        category=AssayCategory.SEQUENCING,
        modalities=["Smart-seq2"],
        description="Plate-based full-length cDNA scRNA-seq",
        typical_duration_days=3.0,
        typical_cost_usd=10000.0,
        requires_live_cells=True,
        throughput="low (96-384 cells)",
        outputs=["fastq", "count_matrix"],
    ),
    "smart-seq3": AssaySpec(
        name="smart-seq3",
        category=AssayCategory.SEQUENCING,
        modalities=["Smart-seq2"],
        description="Improved full-length scRNA-seq with UMIs",
        typical_duration_days=3.0,
        typical_cost_usd=10000.0,
        requires_live_cells=True,
        throughput="low (96-384 cells)",
        outputs=["fastq", "count_matrix"],
    ),
    "bulk_rna_seq": AssaySpec(
        name="bulk_rna_seq",
        category=AssayCategory.SEQUENCING,
        modalities=["bulk_rna_seq"],
        description="Standard bulk RNA sequencing with poly-A or ribo-depletion",
        typical_duration_days=3.0,
        typical_cost_usd=500.0,
        throughput="high",
        outputs=["fastq", "count_matrix"],
    ),
    "atac-seq": AssaySpec(
        name="atac-seq",
        category=AssayCategory.EPIGENOMICS,
        modalities=["scATAC-seq"],
        description="Assay for Transposase-Accessible Chromatin using sequencing",
        typical_duration_days=2.0,
        typical_cost_usd=6000.0,
        requires_live_cells=True,
        outputs=["fastq", "fragments", "peak_matrix"],
    ),
    "cite-seq": AssaySpec(
        name="cite-seq",
        category=AssayCategory.PROTEOMICS,
        modalities=["CITE-seq"],
        description="Simultaneous RNA + surface protein via DNA-barcoded antibodies",
        typical_duration_days=2.0,
        typical_cost_usd=8000.0,
        requires_live_cells=True,
        throughput="high (5k-20k cells)",
        outputs=["fastq", "count_matrix", "adt_matrix"],
    ),
    "10x_multiome": AssaySpec(
        name="10x_multiome",
        category=AssayCategory.SEQUENCING,
        modalities=["scMultiome"],
        description="Joint scRNA-seq + scATAC-seq from the same cell",
        typical_duration_days=2.0,
        typical_cost_usd=10000.0,
        requires_live_cells=True,
        throughput="high (5k-15k cells)",
        outputs=["fastq", "count_matrix", "fragments"],
    ),
    "visium": AssaySpec(
        name="visium",
        category=AssayCategory.SEQUENCING,
        modalities=["spatial_transcriptomics"],
        description="10x Visium spatially barcoded capture on tissue sections",
        typical_duration_days=3.0,
        typical_cost_usd=7000.0,
        requires_fresh_tissue=True,
        throughput="medium (1k-5k spots)",
        outputs=["fastq", "count_matrix", "spatial_coords", "image"],
    ),
    "visium_hd": AssaySpec(
        name="visium_hd",
        category=AssayCategory.SEQUENCING,
        modalities=["spatial_transcriptomics"],
        description="High-definition Visium with 2µm bin resolution",
        typical_duration_days=3.0,
        typical_cost_usd=10000.0,
        requires_fresh_tissue=True,
        throughput="high",
        outputs=["fastq", "count_matrix", "spatial_coords", "image"],
    ),
    "merfish": AssaySpec(
        name="merfish",
        category=AssayCategory.IMAGING,
        modalities=["MERFISH"],
        description="Multiplexed Error-Robust FISH imaging-based spatial",
        typical_duration_days=5.0,
        typical_cost_usd=20000.0,
        requires_fresh_tissue=True,
        throughput="high (100-1000 genes, millions of transcripts)",
        outputs=["transcript_coords", "cell_segmentation"],
    ),
    "seqfish_plus": AssaySpec(
        name="seqfish_plus",
        category=AssayCategory.IMAGING,
        modalities=["seqFISH"],
        description="Sequential FISH for imaging-based spatial transcriptomics",
        typical_duration_days=5.0,
        typical_cost_usd=15000.0,
        requires_fresh_tissue=True,
        outputs=["transcript_coords"],
    ),
    "slide-seq": AssaySpec(
        name="slide-seq",
        category=AssayCategory.SEQUENCING,
        modalities=["Slide-seq"],
        description="Near-cellular spatial transcriptomics on bead arrays",
        typical_duration_days=3.0,
        typical_cost_usd=8000.0,
        requires_fresh_tissue=True,
        outputs=["count_matrix", "spatial_coords"],
    ),
    "perturb-seq": AssaySpec(
        name="perturb-seq",
        category=AssayCategory.PERTURBATION,
        modalities=["Perturb-seq"],
        description="Pooled CRISPR screen + scRNA-seq readout",
        typical_duration_days=14.0,
        typical_cost_usd=15000.0,
        requires_live_cells=True,
        throughput="high (10k-100k cells)",
        outputs=["fastq", "count_matrix", "guide_assignments"],
    ),
    "crop-seq": AssaySpec(
        name="crop-seq",
        category=AssayCategory.PERTURBATION,
        modalities=["CROP-seq"],
        description="CRISPR dropout screening with scRNA-seq readout",
        typical_duration_days=14.0,
        typical_cost_usd=12000.0,
        requires_live_cells=True,
        throughput="high (10k-50k cells)",
        outputs=["fastq", "count_matrix", "guide_assignments"],
    ),
    "patch-seq": AssaySpec(
        name="patch-seq",
        category=AssayCategory.FUNCTIONAL,
        modalities=["Patch-seq"],
        description="Patch-clamp electrophysiology + scRNA-seq on same neuron",
        typical_duration_days=7.0,
        typical_cost_usd=50000.0,
        requires_live_cells=True,
        throughput="very low (10-100 cells)",
        outputs=["fastq", "count_matrix", "ephys_trace", "morphology"],
    ),
    "sc_hi_c": AssaySpec(
        name="sc_hi_c",
        category=AssayCategory.EPIGENOMICS,
        modalities=["scHi-C"],
        description="Single-cell chromosome conformation capture",
        typical_duration_days=5.0,
        typical_cost_usd=15000.0,
        outputs=["contact_matrix"],
    ),
    "sc_bisulfite": AssaySpec(
        name="sc_bisulfite",
        category=AssayCategory.EPIGENOMICS,
        modalities=["scBS-seq"],
        description="Single-cell bisulfite sequencing for DNA methylation profiling",
        typical_duration_days=5.0,
        typical_cost_usd=12000.0,
        outputs=["methylation_matrix"],
    ),
    "sc_nmt_seq": AssaySpec(
        name="sc_nmt_seq",
        category=AssayCategory.EPIGENOMICS,
        modalities=["scNMT-seq"],
        description="Joint nucleosome occupancy, methylation, and transcription",
        typical_duration_days=7.0,
        typical_cost_usd=25000.0,
        requires_live_cells=True,
        throughput="low (100-1k cells)",
        outputs=["count_matrix", "methylation_matrix", "accessibility_matrix"],
    ),
    "flow_cytometry": AssaySpec(
        name="flow_cytometry",
        category=AssayCategory.FUNCTIONAL,
        modalities=[],
        description="Fluorescence-based cell sorting and phenotyping",
        typical_duration_days=1.0,
        typical_cost_usd=500.0,
        requires_live_cells=True,
        throughput="very high (millions of cells)",
        outputs=["cell_counts", "sorted_cells"],
    ),
    "mass_cytometry_CyTOF": AssaySpec(
        name="mass_cytometry_CyTOF",
        category=AssayCategory.PROTEOMICS,
        modalities=[],
        description="Mass-tag cytometry for 40+ protein markers per cell",
        typical_duration_days=2.0,
        typical_cost_usd=3000.0,
        requires_live_cells=True,
        throughput="high (100k-1M cells)",
        outputs=["protein_expression_matrix"],
    ),
    "western_blot": AssaySpec(
        name="western_blot",
        category=AssayCategory.PROTEOMICS,
        modalities=[],
        description="Protein detection and semi-quantification by size separation",
        typical_duration_days=2.0,
        typical_cost_usd=200.0,
        outputs=["band_image", "relative_quantification"],
    ),
    "qPCR": AssaySpec(
        name="qPCR",
        category=AssayCategory.FUNCTIONAL,
        modalities=[],
        description="Quantitative PCR for targeted gene expression validation",
        typical_duration_days=1.0,
        typical_cost_usd=100.0,
        throughput="low (target genes)",
        outputs=["ct_values", "fold_change"],
    ),
    "immunofluorescence": AssaySpec(
        name="immunofluorescence",
        category=AssayCategory.IMAGING,
        modalities=[],
        description="Antibody-based fluorescence imaging of proteins in situ",
        typical_duration_days=2.0,
        typical_cost_usd=500.0,
        outputs=["fluorescence_image"],
    ),
    "elisa": AssaySpec(
        name="elisa",
        category=AssayCategory.PROTEOMICS,
        modalities=[],
        description="Enzyme-linked immunosorbent assay for secreted protein quantification",
        typical_duration_days=1.0,
        typical_cost_usd=300.0,
        throughput="medium (96-384 well)",
        outputs=["protein_concentration"],
    ),
    "cell_viability_assay": AssaySpec(
        name="cell_viability_assay",
        category=AssayCategory.FUNCTIONAL,
        modalities=[],
        description="MTT/CellTiter-Glo viability and proliferation measurement",
        typical_duration_days=1.0,
        typical_cost_usd=200.0,
        requires_live_cells=True,
        throughput="high (96-384 well)",
        outputs=["viability_scores"],
    ),
}


# ── Registry helper functions ──────────────────────────────────────────────


def tools_for_modality(modality: str) -> List[ToolSpec]:
    """Return all registered tools compatible with a given modality."""
    return [t for t in TOOL_REGISTRY.values() if modality in t.modalities]


def assays_for_modality(modality: str) -> List[AssaySpec]:
    """Return all registered assays that produce a given modality."""
    return [a for a in ASSAY_REGISTRY.values() if modality in a.modalities]


def tools_by_category(category: ToolCategory) -> List[ToolSpec]:
    """Return all registered tools in a given category."""
    return [t for t in TOOL_REGISTRY.values() if t.category == category]


# ── Sub-agents ─────────────────────────────────────────────────────────────


class SubagentType(str, Enum):
    WET_LAB_PLANNER = "wet_lab_planner"
    COMPUTATIONAL_ANALYST = "computational_analyst"
    OMICS_QC_AGENT = "omics_qc_agent"
    CAUSAL_REASONING_AGENT = "causal_reasoning_agent"
    BUDGET_SCHEDULER = "budget_scheduler"
    BIOLOGICAL_RULE_CHECKER = "biological_rule_checker"
    TOOL_EXECUTOR = "tool_executor"
    RETROSPECTIVE_CRITIC = "retrospective_critic"
    REPORT_SYNTHESIZER = "report_synthesizer"
    STATISTICAL_INFERENCE_AGENT = "statistical_inference_agent"
    STUDY_DESIGN_AGENT = "study_design_agent"
    KNOWLEDGEBASE_RETRIEVER = "knowledgebase_retriever"


# ── Action schema ───────────────────────────────────────────────────────────


class ExperimentAction(Action):
    """Structured, compositional action for one experiment / analysis step.

    Hybrid representation: discrete *action_type* plus typed arguments,
    optional sub-agent / tool invocation, and calibration fields.
    """

    action_type: ActionType = Field(
        ...,
        description=(
            "Discrete simulator step type. The environment enforces scientific "
            "prerequisites between steps, so actions should follow a valid "
            "pipeline order."
        ),
    )
    input_targets: List[str] = Field(
        default_factory=list,
        description=(
            "Optional references to prior samples, outputs, or artifacts that "
            "this step consumes."
        ),
    )
    method: Optional[str] = Field(
        None,
        description=(
            "Optional named tool or protocol (for example 'Seurat' or "
            "'CellRanger'). Prefer methods compatible with the current "
            "modality and available tool list because tool choice can change "
            "runtime, cost, and scientific fit."
        ),
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Action-specific settings such as comparison labels, perturbation "
            "targets, or analysis options. Use only parameters that materially "
            "change the scientific step."
        ),
    )
    expected_output_type: Optional[str] = Field(
        None,
        description=(
            "Optional expected artifact or summary that should result from the "
            "step, such as a count matrix, QC report, DE table, or validation "
            "result."
        ),
    )
    justification: Optional[str] = Field(
        None,
        description=(
            "Short scientific rationale explaining why this is the right next "
            "step in the current environment state."
        ),
    )
    invoked_subagent: Optional[SubagentType] = Field(
        None, description="Sub-agent to delegate to, if any"
    )
    tool_call_spec: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Optional structured tool invocation payload when the action needs "
            "a more explicit tool execution plan."
        ),
    )
    confidence: float = Field(
        0.5, ge=0.0, le=1.0, description="Agent confidence in this step"
    )


# ── Intermediate outputs ────────────────────────────────────────────────────


class OutputType(str, Enum):
    QC_METRICS = "qc_metrics"
    COUNT_MATRIX_SUMMARY = "count_matrix_summary"
    EMBEDDING_SUMMARY = "embedding_summary"
    CLUSTER_RESULT = "cluster_result"
    ANNOTATION_RESULT = "annotation_result"
    DE_RESULT = "de_result"
    PATHWAY_RESULT = "pathway_result"
    TRAJECTORY_RESULT = "trajectory_result"
    VALIDATION_RESULT = "validation_result"
    NETWORK_RESULT = "network_result"
    SAMPLE_COLLECTION_RESULT = "sample_collection_result"
    LIBRARY_PREP_RESULT = "library_prep_result"
    SEQUENCING_RESULT = "sequencing_result"
    PERTURBATION_RESULT = "perturbation_result"
    CULTURE_RESULT = "culture_result"
    COHORT_RESULT = "cohort_result"
    FOLLOWUP_DESIGN = "followup_design"
    MARKER_RESULT = "marker_result"
    FAILURE_REPORT = "failure_report"
    SUBAGENT_REPORT = "subagent_report"
    CONCLUSION = "conclusion"
    ANALYSIS_RESULT = "analysis_result"


class IntermediateOutput(BaseModel):
    """A single simulated output from one pipeline step."""

    output_type: OutputType
    step_index: int
    success: bool = True
    quality_score: float = Field(1.0, ge=0.0, le=1.0)
    summary: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
    uncertainty: float = Field(0.0, ge=0.0, le=1.0)
    warnings: List[str] = Field(default_factory=list)
    artifacts_available: List[str] = Field(default_factory=list)


# ── Observable state components ─────────────────────────────────────────────


class ResourceUsage(BaseModel):
    budget_used: float = 0.0
    budget_remaining: float = 100_000.0
    time_used_days: float = 0.0
    time_remaining_days: float = 180.0
    samples_consumed: int = 0
    compute_hours_used: float = 0.0


class PipelineStepRecord(BaseModel):
    step_index: int
    action_type: ActionType
    method: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    output_summary: str = ""
    output_type: OutputType
    success: bool = True
    quality_score: float = 1.0
    resource_cost: float = 0.0
    time_cost_days: float = 0.0


class PaperReference(BaseModel):
    """Metadata for a literature source used to ground a task."""

    title: str
    citation: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    url: Optional[str] = None


class ExpectedFinding(BaseModel):
    """A paper-backed result that the agent should try to recover."""

    finding: str
    category: str = "claim"
    keywords: List[str] = Field(default_factory=list)


class TaskSpec(BaseModel):
    """Specification of the biological problem to solve."""

    problem_statement: str = "Unspecified biological problem"
    modality: str = "scRNA-seq"
    organism: str = "human"
    tissue: str = "blood"
    conditions: List[str] = Field(default_factory=list)
    available_assays: List[str] = Field(
        default_factory=lambda: list(ASSAY_REGISTRY.keys()),
        description=(
            "Assays that are scientifically compatible with this task's "
            "modality. These are the relevant assay choices for the episode, "
            "not an unrestricted catalog."
        ),
    )
    available_tools: List[str] = Field(
        default_factory=lambda: list(TOOL_REGISTRY.keys()),
        description=(
            "Tools filtered to those compatible with the current task "
            "modality. The agent should treat this list as the preferred tool "
            "set for the episode."
        ),
    )
    budget_limit: float = 100_000.0
    time_limit_days: float = 180.0
    prior_observations: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    dataset_metadata: Dict[str, Any] = Field(default_factory=dict)
    paper_references: List[PaperReference] = Field(default_factory=list)
    expected_findings: List[ExpectedFinding] = Field(default_factory=list)
    tags: List[str] = Field(
        default_factory=list,
        description="Scenario tags (e.g. confounders, multicohort) for agent context.",
    )
    difficulty: str = Field(
        default="medium",
        description="Scenario difficulty: easy, medium, or hard.",
    )


class ConclusionClaim(BaseModel):
    claim: str = ""
    top_markers: List[str] = Field(default_factory=list)
    causal_mechanisms: List[str] = Field(default_factory=list)
    predicted_pathways: Dict[str, float] = Field(default_factory=dict)
    evidence_steps: List[int] = Field(default_factory=list)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    claim_type: str = "correlational"
    supporting_data: Dict[str, Any] = Field(default_factory=dict)


# ── Observation schema ──────────────────────────────────────────────────────


class ExperimentObservation(Observation):
    """Full observable state returned to the agent at each timestep.

    Deliberately excludes hidden latent biological truth, hidden failure
    conditions, and ground-truth mechanisms.
    """

    task: TaskSpec = Field(default_factory=TaskSpec)
    step_index: int = 0
    pipeline_history: List[PipelineStepRecord] = Field(default_factory=list)
    available_assays: List[str] = Field(
        default_factory=list,
        description=(
            "Episode-specific assay choices already filtered to the current "
            "modality and task context."
        ),
    )
    available_tools: List[str] = Field(
        default_factory=list,
        description=(
            "Episode-specific compatible tools. These are the methods the "
            "agent should prefer instead of inventing incompatible tools."
        ),
    )
    resource_usage: ResourceUsage = Field(
        default_factory=ResourceUsage,
        description=(
            "Running budget, time, and compute usage after previous actions."
        ),
    )
    latest_output: Optional[IntermediateOutput] = None
    all_outputs: List[IntermediateOutput] = Field(default_factory=list)
    discovered_markers: List[str] = Field(default_factory=list)
    candidate_mechanisms: List[str] = Field(default_factory=list)
    uncertainty_summary: Dict[str, float] = Field(default_factory=dict)
    subagent_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    conclusions: List[ConclusionClaim] = Field(default_factory=list)
    rule_violations: List[str] = Field(default_factory=list)
    step_reward_breakdown: Dict[str, float] = Field(default_factory=dict)


AGENT_ACTION_GUIDANCE: Dict[ActionType, str] = {
    ActionType.COLLECT_SAMPLE: (
        "Wet-lab entry point. One successful collection usually provides enough "
        "material to continue unless the output shows poor yield or quality."
    ),
    ActionType.SELECT_COHORT: (
        "Use when subject stratification is part of the scientific question "
        "before downstream experimental work."
    ),
    ActionType.PREPARE_LIBRARY: (
        "Requires collected samples and converts biological material into "
        "sequence-ready libraries."
    ),
    ActionType.CULTURE_CELLS: (
        "Requires collected samples and adds substantial time; use only when "
        "live-cell expansion or later perturbation is needed."
    ),
    ActionType.PERTURB_GENE: (
        "Requires samples. Use for causal tests, not as a default discovery "
        "step."
    ),
    ActionType.PERTURB_COMPOUND: (
        "Requires samples. Best for mechanistic follow-up or treatment "
        "response questions."
    ),
    ActionType.SEQUENCE_CELLS: (
        "Requires prepared libraries and produces the raw sequencing-derived "
        "artifacts used by downstream QC and analysis."
    ),
    ActionType.RUN_QC: (
        "Requires sequencing and returns summarized quality metrics such as "
        "doublets, mitochondrial fraction, and ambient RNA."
    ),
    ActionType.FILTER_DATA: (
        "Requires QC and removes poor-quality cells, changing downstream cell "
        "counts and data retention."
    ),
    ActionType.NORMALIZE_DATA: (
        "Requires filtered data and unlocks clustering, differential "
        "expression, trajectory, and network analyses."
    ),
    ActionType.INTEGRATE_BATCHES: (
        "Requires normalized data. Use when batch effects are likely to "
        "confound interpretation; it is not always necessary."
    ),
    ActionType.CLUSTER_CELLS: (
        "Requires normalized data and identifies cell populations or states "
        "for downstream interpretation."
    ),
    ActionType.ANNOTATE_CELL_TYPES: (
        "Requires clustering. Assigns biological identities to clusters using "
        "marker expression or reference; improves interpretability of DE and "
        "conclusions."
    ),
    ActionType.DIFFERENTIAL_EXPRESSION: (
        "Requires normalized data and is the main route to candidate genes "
        "for pathway analysis and marker selection."
    ),
    ActionType.TRAJECTORY_ANALYSIS: (
        "Requires normalized data and is most useful when lineage progression "
        "or pseudotime is central to the task."
    ),
    ActionType.PATHWAY_ENRICHMENT: (
        "Requires differential expression. Results are less reliable without a "
        "strong DE gene list."
    ),
    ActionType.REGULATORY_NETWORK_INFERENCE: (
        "Requires normalized data and is most helpful once cell states or "
        "trajectories are already characterized."
    ),
    ActionType.MARKER_SELECTION: (
        "Requires differential expression and turns candidate genes into a "
        "short list for validation."
    ),
    ActionType.VALIDATE_MARKER: (
        "Requires discovered markers and is an expensive wet-lab confirmation "
        "step that should follow strong computational evidence."
    ),
    ActionType.DESIGN_FOLLOWUP: (
        "Use to propose targeted next experiments once remaining uncertainty "
        "is clear."
    ),
    ActionType.REQUEST_SUBAGENT_REVIEW: (
        "Use for critique or planning support, not as a substitute for "
        "missing experimental evidence."
    ),
    ActionType.SYNTHESIZE_CONCLUSION: (
        "Use once the evidence is sufficient. Do not spend budget on redundant "
        "steps just because more actions are possible."
    ),
    ActionType.ASSESS_CONFOUNDERS: (
        "Check for batch, composition, or covariate confounding before strong "
        "conclusions; requires normalized data."
    ),
    ActionType.STRATIFY_BY_COVARIATE: (
        "Run analysis stratified by a covariate (e.g. batch, sex) to test "
        "robustness; requires prior DE or clustering."
    ),
    ActionType.RUN_SENSITIVITY_ANALYSIS: (
        "Re-run a prior step under different parameters or adjustments and "
        "compare; requires the underlying analysis already done."
    ),
}

ACTION_PARAMETER_GUIDANCE: Dict[ActionType, List[str]] = {
    ActionType.COLLECT_SAMPLE: [
        "sample_source (str, optional): tissue compartment to collect from (for example 'bone_marrow').",
        "n_samples (int, optional): number of donor/sample units to collect.",
        "preservation (str, optional): handling protocol such as 'fresh' or 'cryopreserved'.",
    ],
    ActionType.SELECT_COHORT: [
        "cohort_a (str, required): primary condition/group label.",
        "cohort_b (str, optional): comparator group label.",
        "stratify_by (list[str], optional): covariates like age/sex/batch.",
        "target_n_per_group (int, optional): desired group size.",
    ],
    ActionType.PREPARE_LIBRARY: [
        "chemistry (str, optional): library chemistry (for example '3prime_v3').",
        "target_cells (int, optional): target recovered cells per sample.",
        "indexing_strategy (str, optional): multiplexing/indexing protocol.",
    ],
    ActionType.CULTURE_CELLS: [
        "duration_days (float, optional): planned culture duration.",
        "media_condition (str, optional): media or cytokine context.",
        "density_cells_per_ml (float, optional): seeding density.",
    ],
    ActionType.PERTURB_GENE: [
        "target_gene (str, required): gene symbol to perturb.",
        "perturbation_type (str, optional): knockdown/knockout/overexpression.",
        "delivery_method (str, optional): CRISPRi/lentivirus/electroporation.",
    ],
    ActionType.PERTURB_COMPOUND: [
        "compound_name (str, required): perturbagen name.",
        "dose_uM (float, optional): nominal concentration in micromolar.",
        "duration_hours (float, optional): treatment duration.",
    ],
    ActionType.SEQUENCE_CELLS: [
        "read_depth (int, optional): target reads per cell.",
        "read_length (str, optional): sequencing configuration (for example '28x91').",
        "lane_count (int, optional): number of lanes to allocate.",
    ],
    ActionType.RUN_QC: [
        "min_genes (int, optional): cell filtering lower bound.",
        "max_mito_pct (float, optional): mitochondrial content threshold (0-100).",
        "doublet_strategy (str, optional): method for doublet detection.",
    ],
    ActionType.FILTER_DATA: [
        "min_counts (int, optional): minimum UMI count per cell.",
        "max_counts (int, optional): upper UMI bound to remove outliers.",
        "min_cells_per_gene (int, optional): gene prevalence threshold.",
    ],
    ActionType.NORMALIZE_DATA: [
        "method (str, optional): normalization approach (for example 'log1p').",
        "target_sum (float, optional): scaling target per cell.",
        "regress_out (list[str], optional): confounders to regress.",
    ],
    ActionType.INTEGRATE_BATCHES: [
        "batch_key (str, required): metadata key identifying batches.",
        "integration_method (str, optional): harmony/bbknn/mnn/scvi.",
        "reference_batch (str, optional): anchor batch for integration.",
    ],
    ActionType.CLUSTER_CELLS: [
        "resolution (float, optional): clustering granularity.",
        "n_neighbors (int, optional): neighborhood graph size.",
        "embedding (str, optional): representation such as pca/scvi.",
    ],
    ActionType.ANNOTATE_CELL_TYPES: [
        "method (str, optional): e.g. SingleR, celltypist.",
        "reference (str, optional): reference atlas name.",
    ],
    ActionType.DIFFERENTIAL_EXPRESSION: [
        "comparison (str, required): contrast label (for example 'disease_vs_control').",
        "group_by (str, optional): metadata key defining groups/clusters.",
        "min_logfc (float, optional): minimum effect size threshold.",
    ],
    ActionType.TRAJECTORY_ANALYSIS: [
        "root_population (str, optional): starting state/cell population.",
        "method (str, optional): trajectory method (for example 'paga').",
        "lineage_key (str, optional): subset or lineage selector.",
    ],
    ActionType.PATHWAY_ENRICHMENT: [
        "gene_set_library (str, optional): pathway database name.",
        "ranking_metric (str, optional): statistic used to rank genes.",
        "fdr_threshold (float, optional): significance cutoff.",
    ],
    ActionType.REGULATORY_NETWORK_INFERENCE: [
        "tf_list_source (str, optional): source of transcription-factor priors.",
        "method (str, optional): inference backend (for example 'grnboost2').",
        "min_edge_weight (float, optional): edge confidence cutoff.",
    ],
    ActionType.MARKER_SELECTION: [
        "top_k (int, optional): markers to keep per cluster/contrast.",
        "selection_basis (str, optional): de_auc/de_logfc/specificity.",
        "target_population (str, optional): cluster or label to prioritize.",
    ],
    ActionType.VALIDATE_MARKER: [
        "marker_genes (list[str], required): genes to validate experimentally.",
        "assay (str, optional): validation assay such as 'flow_cytometry'.",
        "replicates (int, optional): experimental replicate count.",
    ],
    ActionType.DESIGN_FOLLOWUP: [
        "objective (str, required): uncertainty or hypothesis to resolve.",
        "proposed_experiments (list[str], optional): concrete follow-up steps.",
        "priority (str, optional): low/medium/high urgency.",
    ],
    ActionType.REQUEST_SUBAGENT_REVIEW: [
        "review_focus (str, required): what to critique (for example 'qc_thresholds').",
        "subagent_type (str, optional): preferred specialist role.",
        "questions (list[str], optional): explicit questions for the reviewer.",
    ],
    ActionType.SYNTHESIZE_CONCLUSION: [
        "claims (list[dict], required): structured claims with markers, mechanisms, pathways, and confidence.",
        "evidence_steps (list[int], optional): supporting pipeline step indices.",
        "open_questions (list[str], optional): unresolved uncertainties.",
    ],
    ActionType.ASSESS_CONFOUNDERS: [
        "candidate_confounders (list[str], optional): variables to test (e.g. batch, sex).",
        "method (str, optional): assessment method (e.g. regression, PCA).",
    ],
    ActionType.STRATIFY_BY_COVARIATE: [
        "covariate (str, required): metadata key to stratify by (e.g. batch, sex).",
        "analysis_type (str, optional): type of stratified analysis (e.g. DE, clustering).",
    ],
    ActionType.RUN_SENSITIVITY_ANALYSIS: [
        "baseline_step_ref (str, optional): reference to the prior step to vary.",
        "variations (list[dict], optional): parameter or adjustment variants to compare.",
    ],
}

AGENT_ENVIRONMENT_RULES: List[str] = [
    (
        "Each successful action already returns summarized scientific evidence, "
        "so repeated sampling or repeated analysis is not the default."
    ),
    (
        "Repeat a step only when the task demands it or when prior outputs show "
        "poor quality, insufficient yield, unresolved batch effects, or another "
        "clear failure mode."
    ),
    (
        "The available tool and assay lists are already filtered to the current "
        "task modality, so prefer them over inventing incompatible methods."
    ),
    (
        "Hard scientific prerequisites are enforced by the environment, so "
        "invalid pipeline orderings will be blocked."
    ),
]

_TOOL_CATEGORY_AGENT_NOTES: Dict[ToolCategory, str] = {
    ToolCategory.ALIGNMENT: (
        "Best immediately after sequencing to turn FASTQ-like inputs into "
        "count-style matrices for downstream analysis."
    ),
    ToolCategory.PREPROCESSING: (
        "Useful for general single-cell data handling before specialized "
        "downstream analyses."
    ),
    ToolCategory.NORMALIZATION: (
        "Applies after filtering to produce normalized matrices for downstream "
        "modeling."
    ),
    ToolCategory.DIMENSIONALITY_REDUCTION: (
        "Builds latent embeddings that support clustering or trajectory work."
    ),
    ToolCategory.CLUSTERING: (
        "Best once data are normalized and the goal is to resolve cell states "
        "or populations."
    ),
    ToolCategory.DIFFERENTIAL_EXPRESSION: (
        "Tests contrasts and produces ranked genes for biological "
        "interpretation."
    ),
    ToolCategory.TRAJECTORY: (
        "Useful when the task asks about developmental progression, state "
        "transitions, or pseudotime."
    ),
    ToolCategory.GENE_REGULATORY_NETWORK: (
        "Most useful after normalized data and some cell-state structure are "
        "already established."
    ),
    ToolCategory.GENE_SET_ANALYSIS: (
        "Best after differential expression to interpret gene lists at the "
        "pathway level."
    ),
    ToolCategory.BATCH_CORRECTION: (
        "Use when batch effects would confound interpretation; unnecessary use "
        "adds extra steps."
    ),
    ToolCategory.MULTIMODAL_INTEGRATION: (
        "Useful only when combining modalities or batches is part of the "
        "scientific question."
    ),
    ToolCategory.QUALITY_CONTROL: (
        "Helps identify low-quality cells or technical artifacts before "
        "filtering."
    ),
    ToolCategory.CELL_TYPE_ANNOTATION: (
        "Best after clustering when assigning biological identities to groups."
    ),
    ToolCategory.PERTURBATION_ANALYSIS: (
        "Use when perturbations were actually applied and the goal is to model "
        "their transcriptional effects."
    ),
    ToolCategory.SPATIAL: (
        "Only useful when the modality includes spatial coordinates or tissue "
        "context."
    ),
    ToolCategory.STATISTICAL_INFERENCE: (
        "Use for confounder assessment, mixed models, and sensitivity analyses."
    ),
    ToolCategory.KNOWLEDGE_BASE: (
        "Use for reference lookups, gene set metadata, and annotation without compute."
    ),
}


def _format_currency(value: float) -> str:
    return f"${value:,.0f}"


def _format_runtime_hours(hours: float) -> str:
    if hours < 1.0:
        return f"{int(round(hours * 60))}m"
    if float(hours).is_integer():
        return f"{int(hours)}h"
    return f"{hours:.1f}h"


def describe_tool_for_agent(tool_name: str) -> str:
    """Return a compact environment-aware tool description for prompts."""
    tool = TOOL_REGISTRY.get(tool_name)
    if tool is None:
        return tool_name

    parts = [f"{tool.name}: {tool.description}."]
    if tool.input_types or tool.output_types:
        inputs = ", ".join(tool.input_types) or "upstream artifacts"
        outputs = ", ".join(tool.output_types) or "analysis artifacts"
        parts.append(f"Consumes {inputs}; yields {outputs}.")

    category_note = _TOOL_CATEGORY_AGENT_NOTES.get(tool.category)
    if category_note:
        parts.append(category_note)

    resource_bits: List[str] = []
    if tool.typical_cost_usd > 0:
        resource_bits.append(_format_currency(tool.typical_cost_usd))
    if tool.typical_runtime_hours > 0:
        resource_bits.append(_format_runtime_hours(tool.typical_runtime_hours))
    if tool.requires_gpu:
        resource_bits.append("GPU")
    if resource_bits:
        parts.append(f"Typical resources: {', '.join(resource_bits)}.")

    return " ".join(parts)


def describe_assay_for_agent(assay_name: str) -> str:
    """Return a compact environment-aware assay description for prompts."""
    assay = ASSAY_REGISTRY.get(assay_name)
    if assay is None:
        return assay_name

    parts = [f"{assay.name}: {assay.description}."]
    if assay.outputs:
        parts.append(f"Produces {', '.join(assay.outputs)}.")

    requirements: List[str] = []
    if assay.requires_live_cells:
        requirements.append("live cells")
    if assay.requires_fresh_tissue:
        requirements.append("fresh tissue")
    if requirements:
        parts.append(f"Requires {' and '.join(requirements)}.")

    parts.append(
        "Typical resources: "
        f"{_format_currency(assay.typical_cost_usd)}, "
        f"{assay.typical_duration_days:.1f}d."
    )
    return " ".join(parts)


def build_agent_system_prompt() -> str:
    """Build the shared agent system prompt for training and inference."""
    lines = [
        "You are an expert biologist planning a single-cell experiment pipeline.",
        "",
        "Your final goal is to present discovered markers and causal mechanisms in the conclusion (synthesize_conclusion step).",
        "",
        "At each turn you see the experiment state and must pick the next scientifically justified step.",
        "",
        "Environment-specific reasoning rules:",
    ]
    lines.extend(f"  - {rule}" for rule in AGENT_ENVIRONMENT_RULES)
    lines.append("")
    lines.append("Action guidance:")
    lines.extend(
        f"  - {action_type.value}: {AGENT_ACTION_GUIDANCE[action_type]}"
        for action_type in ActionType
    )
    lines.append("")
    lines.append("Action parameter contract (use these parameter keys/types):")
    for action_type in ActionType:
        lines.append(f"  - {action_type.value}:")
        for field_doc in ACTION_PARAMETER_GUIDANCE[action_type]:
            lines.append(f"      * {field_doc}")
    lines.extend([
        "",
        "Respond with ONLY valid JSON, nothing else:",
        '{"action_type": "...", "method": null, "parameters": {}, "justification": "...", "confidence": 0.8}',
        "",
        "For synthesize_conclusion, use structured claims:",
        '{"action_type": "synthesize_conclusion", "parameters": {"claims": [{"top_markers": ["GENE1", "GENE2"], "causal_mechanisms": ["mechanism description"], "predicted_pathways": {"pathway_name": 0.8}, "confidence": 0.8, "claim_type": "causal", "claim": "optional free text"}]}, "justification": "...", "confidence": 0.8}',
    ])
    return "\n".join(lines)


def build_agent_observation_context(
    obs: ExperimentObservation,
    *,
    max_tools: int = 6,
    max_assays: int = 3,
) -> str:
    """Summarize modality-specific tool and assay context for the agent."""
    sections: List[str] = []

    if obs.task.tags or obs.task.difficulty != "medium":
        parts = []
        if obs.task.tags:
            parts.append(f"Task tags: {', '.join(obs.task.tags)}")
        if obs.task.difficulty != "medium":
            parts.append(f"Difficulty: {obs.task.difficulty}")
        sections.append(". ".join(parts) + ".")

    modality_spec = MODALITY_REGISTRY.get(obs.task.modality)
    if modality_spec is not None:
        sections.append(
            "Modality context: "
            f"{modality_spec.name} measures {modality_spec.measurement} at "
            f"{modality_spec.resolution} resolution; typical scale "
            f"{modality_spec.typical_cells}."
        )
    else:
        sections.append(f"Modality context: {obs.task.modality}.")

    tool_names = list(dict.fromkeys(obs.available_tools or obs.task.available_tools))
    if tool_names:
        sections.append("Available tools (already filtered to this modality):")
        for tool_name in tool_names[:max_tools]:
            sections.append(f"  - {describe_tool_for_agent(tool_name)}")
        if len(tool_names) > max_tools:
            remainder = ", ".join(tool_names[max_tools:max_tools + 6])
            sections.append(
                "  - Additional compatible tools not shown in full: "
                f"{remainder}"
            )

    assay_names = list(dict.fromkeys(obs.available_assays or obs.task.available_assays))
    if assay_names:
        sections.append("Available assays:")
        for assay_name in assay_names[:max_assays]:
            sections.append(f"  - {describe_assay_for_agent(assay_name)}")
        if len(assay_names) > max_assays:
            remainder = ", ".join(assay_names[max_assays:max_assays + 4])
            sections.append(
                "  - Additional compatible assays not shown in full: "
                f"{remainder}"
            )

    return "\n".join(sections)
