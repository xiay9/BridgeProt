# BridgeProt: Bridging Free-Form Generation and Explicit Pair Prediction for Emotion-Cause Pair Extraction

This repository will be released publicly, including code and data.

## Overview

**BridgeProt** is a structured generative framework for **Emotion-Cause Pair Extraction (ECPE)** in conversations.

Emotion-Cause Pair Extraction aims to identify emotional utterances and their corresponding cause utterances from dialogue turns. Existing generative models often produce free-form natural language responses, which are flexible but difficult to align with the explicit turn-indexed emotion-cause pairs required by standard ECPE evaluation.

BridgeProt addresses this mismatch by converting free-form generation into explicit structured pair prediction. Each prediction is represented as a normalized record containing an emotion turn and a sorted, duplicate-free list of cause turns.

A typical BridgeProt output is:

```json
{
  "records": [
    {
      "emotion_turn": 7,
      "evidence": [5, 6]
    }
  ]
}
```

This design preserves the flexibility of generative models while ensuring that the output can be directly parsed, scored, and evaluated as explicit ECPE decisions.

---

## Table of Contents

- [Overview](#overview)
- [Method](#method)
- [Framework](#framework)
- [Datasets](#datasets)
- [Main Results](#main-results)
- [Training Regime Results](#training-regime-results)
- [Ablation Study](#ablation-study)
- [Output Format](#output-format)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

---

## Method

BridgeProt consists of three major stages:

### 1. Dialogue-Level Proposal

The model first reads the serialized dialogue and generates candidate emotion-cause records.  
Unlike free-form generation, the proposal is already expressed in the final structured format used for evaluation.

Each record contains:

- `emotion_turn`: the dialogue turn where the emotion appears
- `evidence`: the cause turn or turns supporting the emotion

### 2. Emotion-Centered Verification

For each proposed emotion turn, BridgeProt builds a local verification context and checks whether the proposed cause turns should be retained, revised, or rejected.

This stage helps the model distinguish true causal evidence from nearby but irrelevant dialogue context.

### 3. Structured Reconstruction

After verification, BridgeProt reconstructs the final output into a normalized JSON object.

The reconstruction process guarantees:

- valid JSON structure
- ascending order of emotion turns
- sorted and duplicate-free cause lists
- direct conversion to emotion-cause pair sets
- stable evaluation under ECPE metrics

BridgeProt also supports an optional `explanation` field, but the decision-only format is recommended as the default setting because it is more reliable for official ECPE scoring.

---

## Framework

The overall BridgeProt framework includes:

1. multimodal dialogue serialization
2. structured proposal generation
3. emotion-centered local verification
4. decision fusion
5. schema-preserving reconstruction

![BridgeProt Framework](figures/framework.png)

---

## Datasets

Experiments are conducted on three public multimodal ECPE benchmarks:

| Dataset | Language | Modalities | Description |
|---|---|---|---|
| ECF | English | Text, Audio, Video | English conversational ECPE benchmark |
| MEC4 | Chinese | Text, Audio, Video | Chinese multimodal ECPE benchmark |
| MECAD | Chinese | Text, Audio, Video | Multi-scenario Chinese ECPE benchmark |

Four modality settings are evaluated:

| Setting | Description |
|---|---|
| T | Text only |
| T+A | Text + Audio |
| T+V | Text + Video |
| T+A+V | Text + Audio + Video |

---

## Main Results

### Comparison with Published ECPE Systems

| Dataset | System | Best Modality | Emotion F1 | Cause F1 | Pair F1 |
|---|---|---:|---:|---:|---:|
| ECF | MECPE-2steps | T+A | 79.17 | 70.27 | 53.48 |
| ECF | HiLo | T+A+V | **79.35** | 72.28 | **55.45** |
| ECF | BridgeProt | T+A | 73.94 | **72.35** | 54.55 |
| MEC4 | MECPE-2steps | T+A | — | — | 28.82 |
| MEC4 | HiLo | T+A+V | — | — | 35.11 |
| MEC4 | M³F | T+A+V | — | — | **44.79** |
| MEC4 | BridgeProt | T+A+V | **67.23** | **58.63** | 42.65 |
| MECAD | SHARK | T | 68.14 | 65.24 | 45.74 |
| MECAD | M³HG | T | 70.02 | 68.12 | 50.27 |
| MECAD | BridgeProt | T+A | **77.54** | **74.67** | **57.89** |

BridgeProt achieves competitive performance across all three benchmarks.  
On MECAD, BridgeProt obtains the best Emotion F1, Cause F1, and Pair F1 among the compared systems.

---

## Ablation Study

### Free-Form Generation vs. BridgeProt

BridgeProt significantly outperforms matched free-form generation under the same backbone, training regime, and evaluation setting.

| Dataset | Setting | Free-form Pair F1 | BridgeProt Pair F1 | BridgeProt Validity |
|---|---|---:|---:|---:|
| ECF | 0-shot | 0.11 | 26.25 | 100.00 |
| ECF | 3-shot | 0.52 | 41.10 | 100.00 |
| ECF | LoRA | 1.88 | 50.76 | 100.00 |
| ECF | SFT | 1.37 | 52.77 | 100.00 |
| MEC4 | 0-shot | 0.56 | 14.54 | 100.00 |
| MEC4 | 3-shot | 0.82 | 18.78 | 100.00 |
| MEC4 | LoRA | 0.47 | 40.05 | 100.00 |
| MEC4 | SFT | 2.37 | 42.65 | 100.00 |
| MECAD | 0-shot | 0.45 | 31.14 | 100.00 |
| MECAD | 3-shot | 6.72 | 40.47 | 100.00 |
| MECAD | LoRA | 5.08 | 47.07 | 100.00 |
| MECAD | SFT | 2.97 | 50.50 | 100.00 |

These results show that the main improvement comes from aligning generation with the explicit emotion-cause pair object required by ECPE evaluation.

### Effect of Explanation Field

BridgeProt can optionally include an auxiliary `explanation` field:

```json
{
  "records": [
    {
      "emotion_turn": 7,
      "evidence": [5, 6],
      "explanation": "Turns 5 and 6 provide the immediate trigger for the emotional reaction in turn 7."
    }
  ]
}
```

However, experiments show that the explanation field only provides limited benefits in some 0-shot settings and usually weakens performance under stronger supervision. Therefore, the decision-only format is recommended as the default output format.

### Structural Validity

BridgeProt maintains **100% structural validity** in all main settings, meaning every prediction can be directly parsed and evaluated as explicit emotion-cause pairs.

---

## Output Format

The default BridgeProt output format is:

```json
{
  "records": [
    {
      "emotion_turn": 3,
      "evidence": [1, 2]
    },
    {
      "emotion_turn": 7,
      "evidence": [5, 6]
    }
  ]
}
```

Only the following fields are used for official ECPE scoring:

- `emotion_turn`
- `evidence`

The optional `explanation` field is not used for official scoring.

---

## Repository Structure

```text
BridgeProt/
├── README.md
├── figures/
│   └── framework.png
├── data/
│   ├── ECF/
│   ├── MEC4/
│   └── MECAD/
├── prompts/
│   ├── proposal_prompt.txt
│   ├── verification_prompt.txt
│   └── judge_prompt.txt
├── src/
│   ├── serialize_dialogue.py
│   ├── proposal_generation.py
│   ├── local_verification.py
│   ├── reconstruction.py
│   └── evaluation.py
├── scripts/
│   ├── run_zero_shot.sh
│   ├── run_three_shot.sh
│   ├── run_lora.sh
│   └── run_sft.sh
└── results/
    ├── ecf_results.json
    ├── mec4_results.json
    └── mecad_results.json
```

