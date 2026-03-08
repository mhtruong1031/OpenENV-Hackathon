---
title: Bio Experiment Environment Server
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - bioinformatics
---

# Bio Experiment Environment

An OpenEnv-compatible reinforcement learning environment for planning single-cell RNA-seq (and related) biological experiment pipelines. The agent plans step-by-step without ever observing the true biological state, making this a fully-specified **partially observable Markov decision process (POMDP)**.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Core Concepts](#core-concepts)
   - [Hidden State](#hidden-state)
   - [Visible State](#visible-state)
   - [Episode Lifecycle](#episode-lifecycle)
6. [Action Space](#action-space)
7. [Observation Space](#observation-space)
8. [Reward System](#reward-system)
9. [Rule Engine](#rule-engine)
10. [Simulator Internals](#simulator-internals)
11. [Scenarios](#scenarios)
12. [Interfaces](#interfaces)
    - [In-Process Python](#1-in-process-python)
    - [HTTP / WebSocket Server](#2-http--websocket-server)
    - [Gymnasium Wrapper](#3-gymnasium-wrapper)
    - [LLM Planner Runner](#4-llm-planner-runner)
    - [Literature Benchmark](#5-literature-benchmark)
13. [Training & Evaluation](#training--evaluation)
14. [API Reference](#api-reference)
15. [Configuration Reference](#configuration-reference)
16. [Testing](#testing)
17. [Design Rationale](#design-rationale)

---

## Overview

The environment simulates a scientific team planning a single-cell transcriptomics experiment. The **hidden state** contains the real biology (true DE genes, cell populations, regulatory networks, technical noise parameters). The **agent** only receives noisy simulated outputs and must sequence experiment and analysis steps in a scientifically valid order, manage a budget and timeline, and ultimately synthesize a conclusion that is scored against hidden ground truth.

This setup is useful for:

- Agent planning benchmarks on long-horizon scientific reasoning tasks
- Reinforcement learning with sparse scientific feedback
- Literature-grounded evaluation of LLM planners
- Comparing structured policies against model-free baselines

---

## Project Structure

```
hackathon/
├── __init__.py
├── models.py                          # Shared Pydantic schemas (actions, observations, tasks)
├── client.py                          # OpenEnv HTTP/WebSocket client
├── run_agent.py                       # LLM planner runner (Qwen model)
├── pyproject.toml                     # Package config and dependencies
├── uv.lock                            # Reproducible lockfile (uv)
├── outputs/                           # Episode output files (.gitkeep)
│
├── server/
│   ├── app.py                         # FastAPI server entry point
│   ├── hackathon_environment.py       # Main orchestration (BioExperimentEnvironment)
│   ├── requirements.txt               # Server-specific pip requirements
│   │
│   ├── simulator/
│   │   ├── latent_state.py            # Hidden biological/technical/progress/resource state
│   │   ├── noise.py                   # Reproducible stochastic noise model
│   │   ├── output_generator.py        # Simulated outputs conditioned on hidden state
│   │   └── transition.py             # Transition dynamics, action costs, artifact propagation
│   │
│   ├── rules/
│   │   └── engine.py                  # Hard/soft rule constraint checker
│   │
│   ├── rewards/
│   │   └── reward.py                  # Decomposable step + terminal reward
│   │
│   └── tasks/
│       ├── scenarios.py               # Curated biological scenario library (4 scenarios)
│       └── generator.py              # Task + latent-state generator with domain randomisation
│
├── training/
│   ├── gym_wrapper.py                 # Gymnasium-compatible wrapper
│   ├── evaluation.py                  # Evaluation metrics suite
│   ├── trajectory.py                  # Trajectory serialisation and dataset utilities
│   └── literature_benchmark.py        # Paper-grounded benchmark runner
│
└── tests/
    ├── test_environment.py
    ├── test_simulator.py
    ├── test_rewards.py
    ├── test_rules.py
    ├── test_models.py
    └── test_literature_benchmark.py
```

---

## Installation

The project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install all packages including dev extras
uv sync --extra dev

# Install only core + training extras
uv sync --extra train

# Install with bioinformatics libraries (scanpy, gseapy, biopython)
uv sync --extra bio
```

**Core dependencies** (`pyproject.toml`):

| Package | Version |
|---|---|
| `openenv-core[core]` | `>=0.2.0` |
| `numpy` | `>=1.24.0` |
| `scipy` | `>=1.10.0` |
| `pydantic` | `>=2.0.0` |

**Optional extras:**

| Extra | Packages |
|---|---|
| `train` | `gymnasium>=0.29.0` |
| `bio` | `biopython>=1.84`, `gseapy>=1.1.3`, `scanpy>=1.10.0` |
| `dev` | `pytest>=8.0.0`, `pytest-cov>=4.0.0`, `gymnasium>=0.29.0` |

**Server** (`server/requirements.txt`):

```
openenv[core]>=0.2.0
fastapi>=0.115.0
uvicorn>=0.24.0
```

---

## Quick Start

### In-process

```python
from models import ActionType, ExperimentAction
from server.hackathon_environment import BioExperimentEnvironment

env = BioExperimentEnvironment(scenario_name="cardiac_disease_de")
obs = env.reset()

print(obs.task.problem_statement)

obs = env.step(ExperimentAction(
    action_type=ActionType.COLLECT_SAMPLE,
    parameters={"n_samples": 8},
    justification="Collect enough material for downstream single-cell analysis.",
    confidence=0.9,
))

print(obs.latest_output.summary)
print(f"Reward: {obs.reward:.3f}")
print(f"Budget remaining: ${obs.resource_usage.budget_remaining:,.0f}")
```

### Server mode

```bash
uv run uvicorn server.app:app --reload
# Listening on http://localhost:8000
```

```python
from client import BioExperimentEnv
from models import ActionType, ExperimentAction

with BioExperimentEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    result = env.step(ExperimentAction(action_type=ActionType.COLLECT_SAMPLE))
    print(result.observation.latest_output.summary)
```

### Run the LLM planner

```bash
python run_agent.py
```

### Run the literature benchmark

```bash
python -m training.literature_benchmark \
  --problem-statement "Identify differentially expressed genes in dilated cardiomyopathy" \
  --scenario-name cardiac_disease_de \
  --json
```

---

## Core Concepts

### Hidden State

The simulator maintains a `FullLatentState` that the agent never directly sees:

```
FullLatentState
├── biology: LatentBiologicalState
│   ├── cell_populations        # True cell types with proportions and markers
│   ├── true_de_genes           # Comparison → {gene: log2FC}
│   ├── true_pathways           # Pathway → activity level
│   ├── gene_programs           # Latent transcriptional programmes
│   ├── true_trajectory         # Pseudotime / lineage structure
│   ├── true_regulatory_network # TF → target genes
│   ├── perturbation_effects    # Target → {gene: effect_size}
│   ├── confounders             # Confounding variables
│   ├── true_markers            # Ground-truth marker genes
│   ├── causal_mechanisms       # Ground-truth causal descriptions
│   └── n_true_cells            # (default 10,000)
│
├── technical: TechnicalState
│   ├── batch_effects           # Batch → severity
│   ├── ambient_rna_fraction    # (default 0.05)
│   ├── doublet_rate            # (default 0.04)
│   ├── dropout_rate            # (default 0.10)
│   ├── sample_quality          # (default 0.90)
│   ├── library_complexity      # (default 0.80)
│   ├── sequencing_depth_factor # (default 1.00)
│   └── capture_efficiency      # (default 0.60)
│
├── progress: ExperimentProgress
│   └── (18 boolean milestone flags + 4 count fields)
│
└── resources: ResourceState
    ├── budget_total            # (default $100,000)
    ├── budget_used
    ├── time_limit_days         # (default 180)
    ├── time_used_days
    ├── samples_available / consumed
    ├── compute_hours_used
    ├── sequencing_lanes_used
    └── reagent_kits_used
```

### Visible State

The agent receives `ExperimentObservation` at every step:

```
ExperimentObservation
├── task: TaskSpec              # Problem statement, budget, time limit, assays, tools
├── step_index                  # Current step number
├── pipeline_history            # List of PipelineStepRecord (action summaries)
├── available_assays            # e.g. ["10x_chromium", "smart-seq2", "atac-seq"]
├── available_tools             # e.g. ["CellRanger", "Seurat", "Scanpy"]
├── resource_usage              # Budget and time remaining (not hidden totals)
├── latest_output               # IntermediateOutput from the most recent step
├── all_outputs                 # All outputs accumulated so far
├── discovered_markers          # Markers found via marker_selection
├── candidate_mechanisms        # Regulators/pathways found so far
├── uncertainty_summary         # Avg quality and uncertainty over last 5 outputs
├── subagent_outputs            # Reports from delegated subagents
├── conclusions                 # ConclusionClaims if synthesize_conclusion was called
├── rule_violations             # Active violation messages
├── step_reward_breakdown       # Decomposed reward components
├── reward                      # Scalar step reward
└── done                        # Whether the episode is over
```

### Episode Lifecycle

```
env.reset()
  └─ generate UUID seed
  └─ reseed noise model
  └─ TaskGenerator.generate(seed, scenario_name)
       ├─ pick scenario (random or by name)
       ├─ optionally domain-randomise parameters
       └─ build FullLatentState (hidden) + TaskSpec (visible)
  └─ clear history, outputs, discoveries, conclusions, cumulative reward
  └─ return ExperimentObservation (step_index=0)

for each agent action:
  env.step(action)
    1.  Increment step counter
    2.  Deep-copy previous latent state (for reward shaping comparison)
    3.  RuleEngine.check(action, latent) → hard violations, soft violations
    4.  If hard violation → return FAILURE_REPORT (no resource deduction)
    5.  Deduct budget + time per ACTION_COSTS table
    6.  If resources exhausted → return FAILURE_REPORT, done=True
    7.  Update ExperimentProgress milestone flags
    8.  OutputGenerator.generate(action, latent_state, step_idx)
    9.  If soft violations → quality × 0.5, append warnings to output
   10.  Propagate artifacts (DE genes, cluster names, markers) into latent state
   11.  RewardComputer.step_reward(...) → RewardBreakdown
   12.  Append PipelineStepRecord to history
   13.  If SYNTHESIZE_CONCLUSION → parse ConclusionClaims from parameters
   14.  done = result.done OR step_count >= MAX_STEPS (30)
   15.  If done → RewardComputer.terminal_reward(...) → add to breakdown
   16.  Build and return ExperimentObservation (visible state only)
```

---

## Action Space

There are **21 discrete action types** across three categories.

### Wet-lab actions

| Action | Budget | Time | Description |
|---|---|---|---|
| `collect_sample` | $5,000 | 7 days | Collect biological specimens |
| `select_cohort` | $500 | 1 day | Define sample groups and inclusion criteria |
| `prepare_library` | $8,000 | 3 days | Prepare sequencing library from collected samples |
| `culture_cells` | $3,000 | 14 days | In-vitro cell culture |
| `perturb_gene` | $2,000 | 3 days | Genetic perturbation (knockout, overexpression) |
| `perturb_compound` | $1,000 | 2 days | Chemical/compound perturbation |
| `sequence_cells` | $15,000 | 5 days | Single-cell sequencing run |
| `validate_marker` | $5,000 | 14 days | Experimental validation of a candidate marker |

### Computational actions

Computational actions also add `time_cost_days × 8` compute hours.

| Action | Budget | Time | Description |
|---|---|---|---|
| `run_qc` | $100 | 0.5 days | Quality control metrics and filtering flags |
| `filter_data` | $50 | 0.25 days | Remove low-quality cells/genes |
| `normalize_data` | $50 | 0.25 days | Normalise count matrix and select HVGs |
| `integrate_batches` | $100 | 0.5 days | Batch correction (e.g. Harmony, Scanorama) |
| `cluster_cells` | $100 | 0.5 days | Unsupervised clustering |
| `differential_expression` | $100 | 0.5 days | DE analysis between conditions |
| `trajectory_analysis` | $200 | 1 day | Pseudotime / lineage inference |
| `pathway_enrichment` | $100 | 0.5 days | GSEA or ORA pathway analysis |
| `regulatory_network_inference` | $300 | 1 day | TF-target network reconstruction (e.g. SCENIC) |
| `marker_selection` | $100 | 0.5 days | Select top marker genes per cluster |

### Meta actions (no cost)

| Action | Time | Description |
|---|---|---|
| `design_followup_experiment` | 0.5 days | Propose the next experimental direction |
| `request_subagent_review` | 0.25 days | Delegate a subtask to a specialised sub-agent |
| `synthesize_conclusion` | 0.5 days | Produce final structured conclusion claims (**terminates the episode**) |

### `ExperimentAction` schema

```python
ExperimentAction(
    action_type: ActionType,                    # Required
    input_targets: List[str] = [],              # References to prior outputs/samples
    method: Optional[str] = None,               # Tool name e.g. "Seurat", "Harmony"
    parameters: Dict[str, Any] = {},            # Method-specific params
    expected_output_type: Optional[str] = None, # What the agent expects
    justification: Optional[str] = None,        # Scientific rationale
    invoked_subagent: Optional[SubagentType] = None,
    tool_call_spec: Optional[Dict] = None,      # Structured tool invocation
    confidence: float = 0.5,                    # Agent confidence (0.0–1.0)
)
```

### Subagent types

`wet_lab_planner`, `computational_analyst`, `omics_qc_agent`, `causal_reasoning_agent`, `budget_scheduler`, `biological_rule_checker`, `tool_executor`, `retrospective_critic`, `report_synthesizer`

---

## Observation Space

### `IntermediateOutput` — per-step simulated artifact

| Field | Type | Description |
|---|---|---|
| `output_type` | `OutputType` | One of 19 output types (see below) |
| `step_index` | `int` | Step that produced this output |
| `success` | `bool` | Whether the action succeeded |
| `quality_score` | `float` [0–1] | Output quality, affected by noise and violations |
| `summary` | `str` | Human-readable summary |
| `data` | `Dict[str, Any]` | Type-specific payload (genes, metrics, etc.) |
| `uncertainty` | `float` [0–1] | Epistemic uncertainty estimate |
| `warnings` | `List[str]` | Quality warnings (e.g. high doublet rate) |
| `artifacts_available` | `List[str]` | Named artifacts the agent can reference |

**Output types:** `qc_metrics`, `count_matrix_summary`, `embedding_summary`, `cluster_result`, `de_result`, `pathway_result`, `trajectory_result`, `validation_result`, `network_result`, `sample_collection_result`, `library_prep_result`, `sequencing_result`, `perturbation_result`, `culture_result`, `cohort_result`, `followup_design`, `marker_result`, `failure_report`, `subagent_report`, `conclusion`

### `ResourceUsage` — visible resource state

| Field | Default |
|---|---|
| `budget_used` | `0.0` |
| `budget_remaining` | `$100,000` |
| `time_used_days` | `0.0` |
| `time_remaining_days` | `180` |
| `samples_consumed` | `0` |
| `compute_hours_used` | `0.0` |

### `TaskSpec` — problem specification

| Field | Default |
|---|---|
| `problem_statement` | `"Unspecified biological problem"` |
| `modality` | `"scRNA-seq"` |
| `organism` | `"human"` |
| `tissue` | `"blood"` |
| `conditions` | `[]` |
| `available_assays` | 10x Chromium, Smart-seq2, bulk RNA-seq, ATAC-seq, CITE-seq, Spatial |
| `available_tools` | CellRanger, Seurat, Scanpy, DESeq2, GSEA, Monocle, scVelo, CellChat, SCENIC |
| `budget_limit` | `$100,000` |
| `time_limit_days` | `180` |
| `paper_references` | `[]` |
| `expected_findings` | `[]` |

---

## Reward System

Rewards are fully decomposed for interpretability.

### Step reward formula

```
R_t = r_validity + r_ordering + r_info_gain + r_efficiency + r_novelty + r_penalty + γ[φ(s_{t+1}) − φ(s_t)]
```

| Component | Calculation | Max value |
|---|---|---|
| `validity` | `+0.3` if success; `−1.0` if hard violation | +0.3 |
| `ordering` | `+0.2` if natural next step; `+0.06` if other valid step | +0.2 |
| `info_gain` | `0.4 × quality_score × (1 − uncertainty)` | +0.4 |
| `efficiency` | `0.3 × max(0, 1 − 5 × budget_fraction_used)` | +0.3 |
| `novelty` | `+0.1` if no soft violations | +0.1 |
| `penalty` | `−0.50` per hard violation; `−0.15` per soft violation | — |
| `shaping` | `γ × φ(next) − φ(prev)` where `γ = 0.99` | — |

**Progress potential** `φ(s)` counts completed milestones / 12. Milestones: `samples_collected`, `library_prepared`, `cells_sequenced`, `qc_performed`, `data_filtered`, `data_normalized`, `cells_clustered`, `de_performed`, `pathways_analyzed`, `markers_discovered`, `markers_validated`, `conclusion_reached`.

### Terminal reward formula

Added once when the episode ends:

```
R_T = 3.0 × completeness + 4.0 × calibration + 1.0 × (budget_eff + time_eff) / 2 + overconfidence_penalty
```

| Component | Calculation |
|---|---|
| `completeness` (×3.0) | Fraction of 7 core milestones completed |
| `calibration` (×4.0) | Fraction of conclusions mentioning true mechanisms/markers minus 0.3 for misses |
| `resource_efficiency` (×1.0) | Average of remaining budget and time fractions |
| `overconfidence_penalty` | `−0.5 × confidence` per high-confidence (>0.8) incorrect claim |

The calibration component is the dominant signal — the agent must produce accurate, evidence-based conclusions to achieve high total reward.

### `RewardBreakdown` fields

`validity`, `ordering`, `info_gain`, `efficiency`, `novelty`, `penalty`, `shaping`, `terminal` — all floats, summed in the `total` property.

---

## Rule Engine

The `RuleEngine` enforces scientific constraints before each action. Violations are split into **hard** (block the action entirely) and **soft** (degrade output quality by 50% and add penalty).

### Prerequisite rules (HARD)

| Action | Requires |
|---|---|
| `prepare_library` | `samples_collected` |
| `sequence_cells` | `library_prepared` |
| `run_qc` | `cells_sequenced` |
| `filter_data` | `qc_performed` |
| `normalize_data` | `data_filtered` |
| `integrate_batches` | `data_normalized` |
| `cluster_cells` | `data_normalized` |
| `differential_expression` | `data_normalized` |
| `trajectory_analysis` | `data_normalized` |
| `pathway_enrichment` | `de_performed` |
| `regulatory_network_inference` | `data_normalized` |
| `marker_selection` | `de_performed` |
| `validate_marker` | `markers_discovered` |
| `perturb_gene/compound`, `culture_cells` | `samples_collected` |

### Resource constraints (HARD / SOFT)

- **HARD**: action attempted when budget or time is fully exhausted
- **SOFT**: action cost exceeds remaining budget (allowed but penalised)

### Redundancy rules (SOFT)

Repeating these actions triggers a soft penalty: `collect_sample`, `prepare_library`, `sequence_cells`, `run_qc`, `filter_data`, `normalize_data`

### Causal validity rules (SOFT)

- `synthesize_conclusion` without substantive prior analysis
- Causal claims (`claim_type="causal"`) without validation or network inference evidence
- `pathway_enrichment` before DE analysis

---

## Simulator Internals

### Noise model (`noise.py`)

All randomness is channelled through a single `numpy.Generator` seeded at episode start, making episodes fully reproducible.

Key noise operations:

| Method | Purpose |
|---|---|
| `add_expression_noise` | Per-gene dropout + Gaussian noise |
| `sample_effect_sizes` | Noisy log2FC estimates from true effects |
| `sample_p_values` | P-values via z-statistics using `scipy.stats` |
| `generate_false_positives` | Binomial-sampled spurious DE genes |
| `generate_false_negatives` | True genes missed by analysis |
| `quality_degradation` | Multiply quality by factors, add Gaussian, clip to [0,1] |
| `sample_cluster_count` | Over/under-cluster based on data quality |
| `shuffle_ranking` | Permute a ranking with tunable noise level |

### Output generator (`output_generator.py`)

Each action has a dedicated handler that conditions the output on `FullLatentState`:

| Action → Output | Key noise/conditioning |
|---|---|
| `sequence_cells` → `SEQUENCING_RESULT` | n_cells ~ Poisson(n_true × capture_efficiency); median_UMI ~ Poisson(3000 × depth) |
| `run_qc` → `QC_METRICS` | Doublet/mito/ambient fractions; warns if doublets>8% or mito>10% |
| `cluster_cells` → `CLUSTER_RESULT` | n_clusters from `sample_cluster_count`; Dirichlet cell partition |
| `differential_expression` → `DE_RESULT` | True effects + noise; FP + FN genes; top 50 by \|log2FC\| |
| `trajectory_analysis` → `TRAJECTORY_RESULT` | Quality 0.7 if true trajectory exists else 0.3 |
| `pathway_enrichment` → `PATHWAY_RESULT` | True pathway activities + Gaussian noise; ~2 FP pathways; top 15 |
| `regulatory_network_inference` → `NETWORK_RESULT` | True TF edges + noise edges |
| `marker_selection` → `MARKER_RESULT` | ~20% false-negative rate; ~1% false-positive rate; top 20 |
| `validate_marker` → `VALIDATION_RESULT` | Checks against `true_markers`; 10% chance of incorrect validation |

### Transition engine (`transition.py`)

`TransitionEngine.step` order of operations:

1. Deep-copy state, increment step count
2. If hard violations: return `FAILURE_REPORT` immediately (no cost deducted)
3. Deduct budget and time; add compute hours for computational actions
4. If resources exhausted: return `FAILURE_REPORT`, `done=True`
5. Update `ExperimentProgress` milestone flags
6. Generate `IntermediateOutput` via `OutputGenerator`
7. If soft violations: `quality_score × 0.5`, extend warnings
8. Propagate artifacts (DE genes, cluster names, markers) into latent state
9. Set `done=True` if action is `SYNTHESIZE_CONCLUSION`

### Domain randomisation (`generator.py`)

When `domain_randomise=True` (default), each episode perturbs:

- `budget_limit × Uniform(0.7, 1.3)`
- `time_limit_days × Uniform(0.8, 1.2)`
- `dropout_rate`, `doublet_rate`, `sample_quality`, `ambient_rna_fraction` — small Gaussian noise, clipped
- `batch_effects` values — small Gaussian noise
- Cell `proportion` values — scaled then re-normalised
- `true_de_genes` effect sizes × Uniform(0.8, 1.2)
- `n_true_cells × Uniform(0.6, 1.4)` (minimum 1000)

---

## Scenarios

Four curated scenarios are defined in `server/tasks/scenarios.py`. Pass `scenario_name` to `BioExperimentEnvironment` or `TaskGenerator` to select one explicitly; otherwise a random scenario is chosen each episode.

### `cardiac_disease_de` *(easy)*

| Field | Value |
|---|---|
| Tissue | Heart |
| Conditions | `["healthy", "dilated_cardiomyopathy"]` |
| True markers | `NPPA`, `NPPB`, `POSTN`, `COL1A1` |
| True mechanisms | TGF-beta–driven fibrosis, inflammatory macrophage infiltration |
| Best strategy | DE analysis → pathway enrichment → marker validation |

### `hematopoiesis_trajectory` *(medium)*

| Field | Value |
|---|---|
| Tissue | Bone marrow |
| Conditions | `["steady_state"]` |
| True markers | `GATA1`, `CEBPA`, `SPI1` |
| True mechanisms | GATA1-driven erythroid commitment, PU.1/CEBPA antagonism |
| Best strategy | Trajectory analysis → regulatory network inference → marker selection |

### `perturbation_immune` *(hard)*

| Field | Value |
|---|---|
| Tissue | Synovial fluid |
| Conditions | `["untreated_RA", "JAK_inhibitor_treated"]` |
| True markers | `STAT1`, `SOCS1`, `IFNG` |
| True mechanisms | JAK-STAT inhibition, compensatory Treg expansion |
| Best strategy | Perturbation analysis → DE → pathway enrichment → validate markers |

### `biomarker_validation_lung` *(medium)*

| Field | Value |
|---|---|
| Tissue | Lung |
| Conditions | `["healthy", "IPF"]` |
| True markers | `SPP1`, `MERTK`, `POSTN`, `MMP9` |
| True mechanisms | SPP1+ macrophage-driven fibroblast activation, integrin-mediated SPP1 signalling |
| Best strategy | DE → marker selection → validate_marker (`SPP1`) → pathway enrichment |

---

## Interfaces

### 1. In-Process Python

Use `BioExperimentEnvironment` for direct Python access with full structured observations:

```python
from models import ActionType, ExperimentAction
from server.hackathon_environment import BioExperimentEnvironment

env = BioExperimentEnvironment(
    scenario_name="biomarker_validation_lung",  # None = random
    domain_randomise=True,                       # Perturb parameters each episode
)
obs = env.reset()

# Typical successful pipeline
pipeline = [
    ExperimentAction(action_type=ActionType.COLLECT_SAMPLE,   parameters={"n_samples": 8}),
    ExperimentAction(action_type=ActionType.PREPARE_LIBRARY,  method="10x_chromium"),
    ExperimentAction(action_type=ActionType.SEQUENCE_CELLS,   method="CellRanger"),
    ExperimentAction(action_type=ActionType.RUN_QC,           method="Scanpy"),
    ExperimentAction(action_type=ActionType.FILTER_DATA),
    ExperimentAction(action_type=ActionType.NORMALIZE_DATA,   method="Seurat"),
    ExperimentAction(action_type=ActionType.CLUSTER_CELLS,    parameters={"resolution": 0.5}),
    ExperimentAction(action_type=ActionType.DIFFERENTIAL_EXPRESSION, method="DESeq2"),
    ExperimentAction(action_type=ActionType.PATHWAY_ENRICHMENT, method="GSEA"),
    ExperimentAction(action_type=ActionType.MARKER_SELECTION),
    ExperimentAction(action_type=ActionType.VALIDATE_MARKER,  parameters={"marker": "SPP1"}),
    ExperimentAction(
        action_type=ActionType.SYNTHESIZE_CONCLUSION,
        parameters={
            "claims": [{"claim": "SPP1+ macrophages drive fibrosis in IPF",
                        "confidence": 0.85,
                        "claim_type": "causal",
                        "evidence_steps": [7, 8, 10]}]
        }
    ),
]

for action in pipeline:
    obs = env.step(action)
    if obs.latest_output:
        print(f"[{obs.step_index}] {obs.latest_output.output_type}: {obs.latest_output.summary[:80]}")
    if obs.done:
        print(f"\nFinal reward: {obs.reward:.3f}")
        break
```

### 2. HTTP / WebSocket Server

```bash
uv run uvicorn server.app:app --reload --port 8000
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Reset the environment; returns initial `ExperimentObservation` |
| `POST` | `/step` | Execute an `ExperimentAction`; returns `{observation, reward, done}` |
| `GET` | `/state` | Get current environment state |
| `GET` | `/schema` | Get action and observation JSON schemas |
| `WS` | `/ws` | WebSocket endpoint for persistent sessions |

```python
from client import BioExperimentEnv
from models import ActionType, ExperimentAction

with BioExperimentEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(result.observation.task.problem_statement)

    result = env.step(ExperimentAction(
        action_type=ActionType.COLLECT_SAMPLE,
        parameters={"n_samples": 6},
        confidence=0.8,
    ))
    print(result.observation.latest_output.summary)
    print(f"done={result.done}, reward={result.reward:.3f}")
```

> The default server is configured with `max_concurrent_envs=1`. Increase this in `server/app.py` for parallel evaluation.

### 3. Gymnasium Wrapper

```python
from training.gym_wrapper import BioExperimentGymEnv

env = BioExperimentGymEnv()
obs, info = env.reset()

# Action space: {"action_type": Discrete(21), "confidence": Box(0,1)}
obs, reward, terminated, truncated, info = env.step({
    "action_type": 0,   # ActionType index
    "confidence": 0.8,
})
```

**Observation space** (vectorised):

| Key | Space | Description |
|---|---|---|
| `step_index` | `Discrete(31)` | Current step |
| `budget_remaining_frac` | `Box(0,1)` | Fraction of budget left |
| `time_remaining_frac` | `Box(0,1)` | Fraction of time left |
| `progress_flags` | `MultiBinary(18)` | All 18 ExperimentProgress boolean flags |
| `latest_quality` | `Box(0,1)` | Quality of the most recent output |
| `latest_uncertainty` | `Box(0,1)` | Uncertainty of the most recent output |
| `avg_quality` | `Box(0,1)` | Average quality over all outputs |
| `avg_uncertainty` | `Box(0,1)` | Average uncertainty over all outputs |
| `n_violations` | `Discrete(20)` | Number of current violations |
| `n_outputs` | `Discrete(31)` | Number of outputs produced |
| `cumulative_reward` | `Box(-100, 100)` | Running total reward |

### 4. LLM Planner Runner

`run_agent.py` runs a local Qwen language model as an action planner:

```bash
python run_agent.py
```

**Configuration** (edit `run_agent.py`):

| Constant | Default | Description |
|---|---|---|
| `MODEL_ID` | `"Qwen/Qwen3.5-0.8B"` | HuggingFace model ID |
| `MAX_EPISODE_STEPS` | `12` | Steps before forced termination |

The model is prompted to output a single JSON action:

```json
{
  "action_type": "collect_sample",
  "method": null,
  "parameters": {},
  "justification": "Collect samples to begin the pipeline.",
  "confidence": 0.8
}
```

If the model produces unparseable output, the runner falls back to a hardcoded 11-step pipeline. Device selection prefers Apple Silicon MPS > CUDA > CPU automatically.

### 5. Literature Benchmark

The benchmark runs a paper-aligned action sequence and scores outputs against curated expected findings:

```python
from training.literature_benchmark import run_paper_benchmark

result = run_paper_benchmark(
    problem_statement="Identify differentially expressed genes in dilated cardiomyopathy",
    scenario_name="cardiac_disease_de",   # None = auto-select by keyword match
    domain_randomise=False,
)

print(f"Match ratio: {result.match_ratio:.2%}")
print(f"Matched findings: {result.matched_findings}")
print(f"Missed findings: {result.missed_findings}")
print(f"Final reward: {result.final_reward:.3f}")
```

The benchmark automatically constructs a biologically sensible action sequence:

1. `collect_sample → prepare_library → sequence_cells → run_qc → filter_data → normalize_data → cluster_cells`
2. If the scenario involves trajectories: `trajectory_analysis → regulatory_network_inference → marker_selection`
3. Otherwise: `differential_expression → pathway_enrichment → marker_selection → validate_marker`
4. `synthesize_conclusion` with heuristically inferred claims

**Finding matching:** A finding is "matched" when at least `ceil(n_keywords / 2)` of its keywords appear in the evidence text (output summaries + gene lists + pathway names + discovered mechanisms and markers).

---

## Training & Evaluation

### Trajectory recording (`training/trajectory.py`)

```python
from training.trajectory import Trajectory, TrajectoryDataset

traj = Trajectory(episode_id="ep_001", task=obs.task.model_dump())

for action in pipeline:
    obs = env.step(action)
    traj.add_step(
        action=action.model_dump(),
        observation=obs.model_dump(),
        reward=obs.reward,
        done=obs.done,
        reward_breakdown=obs.step_reward_breakdown,
    )
    if obs.done:
        break

traj.save("outputs/ep_001.json")

# Dataset utilities
dataset = TrajectoryDataset()
dataset.add(traj)
successful = dataset.filter_successful()  # reward > 0 on terminal step
print(dataset.summary())
```

### Evaluation metrics (`training/evaluation.py`)

```python
from training.evaluation import EvaluationSuite

# Online metrics (from trajectory objects)
metrics = EvaluationSuite.online_metrics(trajectories)
# → mean_return, median_return, std_return, mean_episode_length, success_rate

# Benchmark metrics (from trajectory dataset)
metrics = EvaluationSuite.benchmark_metrics(dataset)
# → pipeline_validity_rate, ordering_score, action_diversity, mean_conclusion_confidence

# Simulator fidelity (with real data for comparison)
metrics = EvaluationSuite.simulator_fidelity_metrics(simulated_trajs, real_data=None)
# → reward_distribution_gap (if real data provided)
```

---

## API Reference

### `BioExperimentEnvironment`

```python
class BioExperimentEnvironment:
    MAX_STEPS: int = 30

    def __init__(
        self,
        scenario_name: Optional[str] = None,
        domain_randomise: bool = True,
    ): ...

    def reset(self) -> ExperimentObservation: ...
    def step(self, action: ExperimentAction) -> ExperimentObservation: ...
    def set_scenario(self, scenario_name: str) -> None: ...
```

### `TaskGenerator`

```python
class TaskGenerator:
    def __init__(
        self,
        scenarios: Optional[List[Scenario]] = None,
        domain_randomise: bool = True,
    ): ...

    def generate(
        self,
        *,
        seed: Optional[int] = None,
        scenario_name: Optional[str] = None,
    ) -> Tuple[TaskSpec, FullLatentState]: ...

    def list_scenarios(self) -> List[str]: ...
```

### `RewardComputer`

```python
class RewardComputer:
    def __init__(
        self,
        gamma: float = 0.99,
        efficiency_weight: float = 0.3,
        info_gain_weight: float = 0.4,
        validity_weight: float = 0.3,
    ): ...

    def step_reward(
        self,
        action, prev_state, next_state, output,
        hard_violations, soft_violations,
    ) -> RewardBreakdown: ...

    def terminal_reward(
        self,
        state: FullLatentState,
        conclusions: List[ConclusionClaim],
        task_success_criteria: List[str],
    ) -> RewardBreakdown: ...
```

### `NoiseModel`

```python
class NoiseModel:
    def __init__(self, seed: int = 42): ...
    def reseed(self, seed: int) -> None: ...
    def add_expression_noise(self, true_values, noise_level, dropout_rate) -> Dict: ...
    def sample_effect_sizes(self, true_effects, sample_size, noise_level) -> Dict: ...
    def sample_p_values(self, true_effects, sample_size, noise_level) -> Dict: ...
    def generate_false_positives(self, n_background_genes, fdr) -> List[str]: ...
    def generate_false_negatives(self, true_genes, fnr) -> List[str]: ...
    def quality_degradation(self, base_quality, factors) -> float: ...
    def coin_flip(self, p: float) -> bool: ...
```

---

## Configuration Reference

| Parameter | Location | Default | Description |
|---|---|---|---|
| `MAX_STEPS` | `hackathon_environment.py` | `30` | Max steps per episode |
| `scenario_name` | `BioExperimentEnvironment.__init__` | `None` (random) | Fix a specific scenario |
| `domain_randomise` | `BioExperimentEnvironment.__init__` | `True` | Perturb scenario params each episode |
| `gamma` | `RewardComputer.__init__` | `0.99` | Potential shaping discount |
| `efficiency_weight` | `RewardComputer.__init__` | `0.3` | Weight on resource efficiency |
| `info_gain_weight` | `RewardComputer.__init__` | `0.4` | Weight on information gain |
| `validity_weight` | `RewardComputer.__init__` | `0.3` | Weight on action validity |
| `max_concurrent_envs` | `server/app.py` | `1` | Max simultaneous server sessions |
| `MODEL_ID` | `run_agent.py` | `"Qwen/Qwen3.5-0.8B"` | LLM used as planner |
| `MAX_EPISODE_STEPS` | `run_agent.py` | `12` | Steps for LLM planner episodes |

---

## Testing

```bash
# Full test suite
uv run pytest tests/ -q

# Environment integration tests
uv run pytest tests/test_environment.py -v

# Literature benchmark integration
uv run pytest tests/test_literature_benchmark.py -v

# Sanity-check subset (fast)
uv run pytest tests/test_environment.py tests/test_literature_benchmark.py -q
```

Test coverage:

| File | What it tests |
|---|---|
| `test_environment.py` | Reset/step lifecycle, valid vs invalid pipelines, conclusion termination |
| `test_simulator.py` | Noise model, output generator, transition engine |
| `test_rewards.py` | Step reward components, terminal reward computation |
| `test_rules.py` | Prerequisite checks, redundancy, resource constraints |
| `test_models.py` | Pydantic schema roundtrips and field validation |
| `test_literature_benchmark.py` | Scenario selection, benchmark matching ratio |

---

## Design Rationale

This environment models a realistic scientific planning loop rather than a toy decision problem:

- **Actions have prerequisites** — you cannot sequence before preparing a library. The rule engine enforces a biologically valid partial order.
- **Outputs are noisy and imperfect** — DE results include false positives and negatives. Clustering may over- or under-split. Validation has a 10% error rate.
- **Budget and time are finite** — the total budget of $100,000 and 180-day timeline mean the agent cannot try every action.
- **Calibration is scored, not just success** — an agent that makes high-confidence incorrect claims is penalised. A cautious, evidence-backed conclusion scores better than an overconfident wrong one.
- **Domain randomisation prevents memorisation** — budget, time, noise levels, effect sizes, and cell proportions are perturbed each episode.
- **Rewards are decomposed** — each component (validity, ordering, information gain, efficiency, novelty, terminal) is exposed in `step_reward_breakdown`, making it straightforward to diagnose agent behaviour and tune training.

The environment is suitable for benchmarking LLM planners that must reason about long scientific workflows, training RL agents with curriculum on ordered pipelines, and evaluating how well a system can recover literature-supported findings from noisy simulated data.
