# Translating Radiology Findings into Plain Language: Evaluating FLAN-T5 and Custom PyTorch LoRA on LaymanRRG 

**Author:** Aidan Fassy
**Student Number:** 44368133

## Executive Summary

Radiological reports are written in dense medical jargon that frequently leaves patients anxious, confused, and prone to misinterpretation when reviewing their results on health portals. The objective of this project is to build a sequence-to-sequence NLP prototype on the ACL 2025 BioLaySumm (`LaymanRRG`) benchmark that translates complex diagnostic findings into accurate, layperson-accessible language at approximately a 6th-grade reading level.

To achieve this without the heavy memory overhead of full fine-tuning, we adapt `google/flan-t5-base` (approx. 248M parameters) using a custom PyTorch Low-Rank Adaptation (LoRA) layer injected into the attention query ($W_q$) and value ($W_v$) projections. Freezing 99.64% of the base weights restricts updates to a lightweight rank-8 bottleneck (approx. 884K parameters, 0.36% of total).

However, standard lexical metrics like ROUGE-1, ROUGE-2, and ROUGE-L measure surface $n$-gram overlap rather than clinical truth. A model can subvert these metrics by memorising routine normal findings (e.g., *"lungs are clear, no acute abnormality"*) while omitting critical pathologies or hallucinating non-existent findings. This investigation evaluates whether LoRA fine-tuning genuinely improves factual clinical fidelity over an untuned zero-shot baseline and unconstrained full fine-tuning across both full (`Val_Full`) and deduplicated (`Val_Unseen`) validation splits.

## Architecture & Working Principles

### FLAN-T5: Encoder-Decoder Backbone

![System Architecture](architecture.png)
*Figure 1: Diagram of the FLAN-T5 Model Architecture*

The base model is `google/flan-t5-base` (~248M parameters), consisting of 12 encoder blocks and 12 decoder blocks ($d_{\text{model}} = 768$, 12 attention heads with $d_k = d_v = 64$, feed-forward dimension $d_{\text{ff}} = 2048$, Pre-LN RMSNorm, and relative position biases).

An *Encoder-Decoder* architecture provides a distinct advantage over *decoder-only* causal models (such as GPT-2) for clinical text. Radiological narratives are scattered and interdependent across sections. Unlike left-to-right causal models, FLAN-T5's bidirectional encoder attends to all tokens simultaneously without a causal mask, building a complete representation of the clinical case before generation begins. The autoregressive decoder then acts as a grounded information bottleneck, continually querying the encoder's memory states ($H_{\text{enc}}$) via *cross-attention* so that generated layperson phrasing remains anchored to source observations.

![FLAN-T5 Sequence-to-Sequence Architecture](flan_t5_backbone.png)
*Figure 2: Diagram of the T5 Backbone of the project*

### Custom PyTorch LoRA

#### LoRA Mathematical Breakdown & Implementation Details

In `modules.py`, instead of updating the full $768 \times 768$ linear weight matrices in FLAN-T5, the `LoRALinear` layer factorises the weight updates into two low-rank matrices:

$$h = W_0 x + \frac{\alpha}{r} B A x$$

Here, $W_0 \in \mathbb{R}^{768 \times 768}$ represents the frozen pre-trained weight matrix (`requires_grad = False`), preventing PyTorch from allocating heavy AdamW momentum and variance buffers across all 248M base parameters. 

The parallel adapter path compresses the 768-dimensional token representation through down-projection $A \in \mathbb{R}^{8 \times 768}$ (initialised with Gaussian noise $\sigma = 0.02$ to break symmetry) and maps it back via up-projection $B \in \mathbb{R}^{768 \times 8}$ (initialised to zeros). Because $B = 0$ at step 0, the adapter initially evaluates to zero ($B A x = 0 \implies h = W_0 x$), preserving exact pre-trained representations at the start of training. Scaling by $\frac{\alpha}{r} = \frac{32}{8} = 4.0$ stabilises update magnitudes without retuning the base learning rate. Following Hu et al. [3], a rank of $r = 8$ was selected because task-specific attention adaptation resides in a low intrinsic dimension: $r = 8$ provides sufficient capacity to learn clinical-to-lay translation while keeping trainable weights at $0.36\%$ and acting as a structural bottleneck against the template overfitting that emerges at higher ranks ($r \ge 16$) or full fine-tuning.

#### Injection Targets & Optimiser Memory Analysis
Rather than modifying all projections, `LoRALinear` layers wrap the query ($W_q$) and value ($W_v$) projections across:
* **Encoder Self-Attention:** Steers attention across scattered clinical findings to build a contextual representation.
* **Decoder Self-Attention:** Maintains linguistic fluency, sentence structure, and a layperson tone.
* **Decoder Cross-Attention:** The core translational bridge querying technical details in $H_{\text{enc}}$ to extract plain-language equivalents.

**Base Model Parameters:** $247,577,856$ (frozen, `requires_grad = False`)  
**Trainable LoRA Parameters:** $884,736$ ($0.36\%$ of total)  
**Total Parameters:** $248,462,592$

Because 99.64% of weights are frozen, AdamW only allocates gradient history, momentum ($m_t$), and variance ($v_t$) states for 884K parameters (approx. 7 MB) instead of 248M parameters (approx. 2 GB), reducing peak training VRAM on an A100 GPU from 10.29 GB (Full FT) down to 6.59 GB.

## Data Pipeline & Preprocessing
### Dataset
The `ACL 2025 BioLaySumm Subtask 2.1` dataset was used as a benchmark. This dataset consists of two distinct columns:
* `radiology_report`: This is the source column, which serves as the expert report, written by radiologists.
* `layman_report`: This is the target column. It is a plain-language summary, aimed at around the 6th-grade reading level.

The dataset has split sizes of $150,454$ for training, $10,000$ for validation, and $10,537$ for testing. Because the official shared-task test split is reserved for blind workshop evaluation, and in accordance with the course benchmarking guidelines, all quantitative and clinical safety evaluations are performed on the held-out validation split across both its full (`Val_Full`) and SHA-256 deduplicated (`Val_Unseen`) cohorts.

### Preprocessing & Tokenisation

In `dataset.py`, each input report is prepended with the task prompt `"Translate this radiology report into plain language for the patient: "` to activate FLAN-T5's instruction-following mode. Source narratives are capped at $512$ tokens, which covers over $95\%$ of full clinical reports without truncation, while target summaries are limited to $128$ tokens.

To prevent padding tokens from skewing training, `LaymanRRGDataSet.__getitem__` handles target padding explicitly. While T5 assigns `pad_token_id = 0`, PyTorch's `CrossEntropyLoss` expects `ignore_index = -100` to mask tokens out of the loss calculation:

```python
labels[labels == self._tokeniser.pad_token_id] = -100
```
### Leakage Prevention

A major issue with routine radiology data is that normal screening reports frequently repeat identical template phrasing (e.g., *"Lungs are clear. No acute cardiopulmonary abnormality."*). If these exact templates leak into both the training and validation sets, a model can easily game lexical metrics like ROUGE by simply memorising common phrases rather than learning how to actually translate clinical text.

To catch this, `load_and_split_dataset` normalises each report (lowercasing and stripping punctuation/whitespace) and computes its SHA-256 hash across the $150{,}454$ training cases. The model is then evaluated across two validation cohorts (`--num_samples 500`): `Val_Full`, the official untouched validation split ($10{,}000$ cases) for standard benchmarking, and `Val_Unseen`, a deduplicated subset ($\sim 8{,}600$ unique cases after pruning $\sim 14\%$ exact training-template collisions), allowing us to directly test whether the model is genuinely generalising or just memorising boilerplate.

## Midpoint Feasibility Review

### User Need, Scope & Acceptance Criteria
The intended users for this product are digital health engineering teams developing automated patient portal communication tools to translate dense, technical radiology reports into plain-language summaries at approximately a 6th-grade reading level.

The following acceptance criteria are established for this prototype:
* **Lexical Performance:** Achieve a gain of $\geq 10\%$ in ROUGE-1, ROUGE-2, and ROUGE-L over an untuned zero-shot baseline on the validation split.
* **VRAM Ceiling:** Keep peak GPU memory allocation strictly below $8.0$ GB during LoRA training (batch size $\geq 8$, sequence length 512) and below $4.0$ GB during evaluation on a single GPU node.
* **Clinical Safety Constraint:** Strictly zero hallucinations of acute findings and zero omissions of critical incidental findings (such as an apical pneumothorax or suspicious pulmonary nodule), audited across representative validation cases.
* **Throughput:** Achieve $\geq 30$ tokens/second during inference on Rangpur HPC GPU nodes.
* **Generalisation Check:** Keep the performance drop $\leq 15\%$ on an audited, deduplicated validation split (`Val_Unseen`) compared to the full split (`Val_Full`) to verify genuine generalisation over template memorisation.

### Model Choice & Course Concepts
This project falls under the `[Hard Difficulty]` band, adapting `google/flan-t5-base` (~248M parameters, 32,128-token vocabulary) on the BioLaySumm `LaymanRRG` benchmark. FLAN-T5's bidirectional encoder and cross-attention ground plain-text translations in non-linear source findings, while custom PyTorch LoRA avoids the optimiser memory overhead of full fine-tuning. This setup directly investigates the core course dilemma: preventing models from earning deceptively high ROUGE scores by repeating boilerplate screening templates while failing on acute clinical findings.

### Preliminary Feasibility Evidence
Initial profiling of the `LaymanRRG` dataset confirmed that over $95\%$ of reports fit within the $512$-token limit without cutting off clinical findings. Exact string matching flagged repeated normal screening templates across splits, confirming the need for a deduplicated test set (`Val_Unseen`) to catch pure memorisation. A smoke test on an interactive Rangpur GPU node confirmed that `flan-t5-base` loads and runs autoregressive generation in pure PyTorch with zero issues. Finally, running a forward and backward pass with LoRA on an initial test batch peaked at $\sim 3.4$ GB VRAM on a single GPU. This left ample headroom below the $8.0$ GB ceiling, verifying that training would operate comfortably within our memory budget.

### Risks, Budget & Fallback
Key technical risks and mitigations include:
* **VRAM Out-of-Memory:** Full fine-tuning risks OOM errors due to AdamW optimiser states across 248M parameters on 512-token sequences. Custom LoRA is implemented as the primary strategy to restrict updates to $<0.4\%$ of parameters.
* **Boilerplate Inflation:** High ROUGE scores masking factual hallucinations or omissions of acute pathologies. Addressed through qualitative clinical safety autopsies.
* **Computing Budget:** $\sim 3$ GPU hours for data auditing and pipeline smoke tests; $\sim 8\text{--}10$ GPU hours total for training and benchmarking runs via Slurm on Rangpur.
* **Next Planned Experiment:** Launch a 1-epoch run of the custom PyTorch LoRA training script on Rangpur to verify monotonic loss decrease, tracking peak VRAM allocation and step latency.
* **Workable Fallback:** If memory allocation issues persist under full fine-tuning, reduce batch size to 4 with 4 gradient accumulation steps under FP16/bfloat16. If resource constraints remain intractable, fall back to the $\sim 60\text{M}$ parameter `flan-t5-small` model as the comparative baseline while retaining `flan-t5-base` with LoRA as the primary architecture.

## Experimental Results & Benchmarking

All experiments were executed on an NVIDIA A100-PCIE-40GB GPU using CUDA 12.1, mixed-precision `bfloat16`, a batch size of 8, and beam search decoding ($k=2$). We benchmarked three distinct setups: an untuned Zero-Shot baseline, the custom PyTorch LoRA model ($r=8, \alpha=32$), and a Full Fine-Tuning run. Both trainable configurations were optimised using AdamW (`train.py`), with learning rates set to $\text{LR} = 3 \times 10^{-4}$ for LoRA and $\text{LR} = 3 \times 10^{-5}$ for Full Fine-Tuning—reflecting the standard practice that low-rank adapter bottlenecks require an order-of-magnitude larger step size than unconstrained full-network updates to converge within a single epoch without destabilising pre-trained representations.

### Quantitative Performance & Resource Use

| Configuration | Trainable Params | Total Params | Train / Val Loss | Train Time (1 Epoch) | Peak Train VRAM | Peak Eval VRAM | Eval Throughput | `Val_Full` (R-1 / R-2 / R-L / R-Lsum) | `Val_Unseen` (R-1 / R-2 / R-L / R-Lsum) | Generalisation Drop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot Baseline** | 0 (0.00%) | 247.6M | N/A | N/A | N/A | 1.69 GB | 374.7 tok/s | 0.3493 / 0.1360 / 0.3118 / 0.3121 | 0.3663 / 0.1594 / 0.3188 / 0.3190 | +4.9% (no memorisation) |
| **Custom PyTorch LoRA** | **884,736 (0.36%)** | 248.5M | 1.0451 / 0.7289 | **~44 mins** | **6.59 GB** | **1.69 GB** | **307.8 tok/s** | **0.7066 / 0.5090 / 0.6477 / 0.6484** | **0.6792 / 0.4605 / 0.6087 / 0.6085** | **-3.9% (Target $\le 15\%$)** |
| **Full Fine-Tuning** | 247,577,856 (100.0%) | 247.6M | 0.8299 / 0.5678 | ~63 mins | 10.29 GB | 1.69 GB | 380.3 tok/s | 0.7317 / 0.5474 / 0.6780 / 0.6789 | 0.7040 / 0.4986 / 0.6389 / 0.6392 | -3.8% (Target $\le 15\%$) |

### Training Dynamics
![LoRA Loss Curve](lora_loss.png)
*Figure 3: Training and Validation cross-entropy loss trajectories over $18,806$ steps on an NVIDIA A100 GPU.*

Training converged smoothly over a single epoch, consisting of $18,806$ optimisation steps, using a batch size of $8$. Cross-entropy loss began at roughly $2.75$ on initial batches and steadily decreased to the final training loss of $1.0451$, with the validation loss reaching $0.7289$.

Notably, validation loss remained consistently lower than training loss throughout training. This can be attributed to active regularisation: during training passes, a dropout rate of $p=0.05$ is injected into the LoRA bottleneck to prevent over-reliance on specific token co-occurrences. During evaluation passes, dropout is completely disabled, allowing the full deterministic capacity of the adapted network to be used without injected noise.

The final validation loss of $0.7289$ corresponds to a perplexity of:

$$ e^{0.7289} \approx 2.07$$

Across the vocabulary of $32,128$ tokens, this indicates that the model is, on average, uncertain between barely two candidate tokens at each autoregressive decoding step.

### Takeaways

Despite updating only $0.36\%$ of the network ($884{,}736$ parameters), LoRA captured $96.6\%$ of Full Fine-Tuning's ROUGE-1 ($0.7066$ vs $0.7317$) and $93.0\%$ of its ROUGE-2 ($0.5090$ vs $0.5474$), while cutting peak training VRAM by $36\%$ ($6.59\text{ GB}$ vs $10.29\text{ GB}$) to stay well below our $8.0\text{ GB}$ ceiling. Against the untuned Zero-Shot baseline, LoRA achieved relative gains of $+102.3\%$ in ROUGE-1 and $+274.3\%$ in ROUGE-2, confirming that domain-specific adaptation is essential. Furthermore, evaluating on the deduplicated `Val_Unseen` cohort showed only a $3.9\%$ drop in ROUGE-1 ($0.7066$ to $0.6792$), well within our $\le 15\%$ threshold and proving genuine clinical translation over template memorisation.

## Clinical Safety

In biomedical NLG, standard lexical overlap metrics ($n$-gram ROUGE) are poor proxies for clinical safety: a model can achieve high ROUGE scores by reciting fluent negative templates while hallucinating findings or omitting life-threatening pathologies. Because the untuned Zero-Shot baseline largely echoes truncated fragments of the raw medical prompt ($0.1360$ ROUGE-2), the qualitative audit below focuses on the critical deployment comparison between Custom LoRA and Full Fine-Tuning across three representative validation cases covering jargon simplification, acute airway placement, and layperson readability trade-offs:

| Case # & Focus | Source Report & Gold Standard | Custom LoRA Prediction ($r=8$) | Full Fine-Tuning Prediction | Clinical Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Case 1**<br>*(Jargon Simplification)* | **Source:** Bands of atelectasis and consolidation in the mid-lung fields, along with confluent opacities in the lower mid-lung fields, also showing bilateral involvement. Findings... could correspond to COVID-19-type pneumonic infiltrates. Evaluate progression and correlate with clinical and laboratory data.<br><br>**Gold:** There are areas of lung collapse and thickening in the middle parts of the lungs, along with large blurry spots in the lower middle areas, affecting both lungs... | There are areas of lung collapse and lung thickening in the middle parts of the lungs, along with some cloudy areas in the lower parts of the lungs, which could be due to COVID-19-type pneumonia. It's important to consider the patient's history and laboratory results. | There are areas of lung collapse and solidified lung tissue in the middle and lower parts of the lungs, along with a mix of cloudy areas in the lower parts... It's important to consider the patient's symptoms and lab tests to see if these changes are getting worse. | **Both Pass**<br>Jargon simplified into plain terms; clinical follow-up instructions preserved. |
| **Case 2**<br>*(Airway Safety)* | **Source:** Extensive consolidation in the right hemithorax. Left pleural effusion. Endotracheal tube with the distal tip positioned at the level of the right main bronchus.<br><br>**Gold:** There is a large area of lung tissue that has become solidified on the right side of the chest. There is also fluid buildup around the left lung. The breathing tube is in place, with its tip located at the level of the main airway on the right side. | There is a large area of lung thickening on the right side of the chest. There is fluid buildup on the left side of the chest. A breathing tube is positioned at the level of the right main airway. | There is a large area of lung thickening on the right side of the chest. There is fluid buildup around the left lung. The breathing tube is **correctly positioned** at the level of the main airway on the right side. | **LoRA: Pass** (neutral, faithful to source)<br>**Full FT: Critical Failure** (hallucinated *"correctly positioned"* on tube malposition) |
| **Case 3**<br>*(Readability vs. Hallucination)* | **Source:** Bilateral pulmonary involvement in the form of patchy areas of ground-glass opacity and scattered consolidation... consistent with COVID-19 pneumonia with progression of involvement as of 2 2 3 2 3. No pleural effusion, lymphadenopathy, or other relevant abnormalities are observed.<br><br>**Gold:** Both lungs are affected with irregular areas of haziness and scattered solid patches... typical for COVID-19 pneumonia and seems to be getting worse. There is no fluid around the lungs, swollen lymph nodes, or other noticeable issues. | Both lungs show signs of involvement in the form of **patchy areas of ground-glass opacity and scattered consolidation**, mainly at the edges... with the progression of involvement **as of 2 2 3 2 3**. There is no fluid around the lungs, lymph nodes, or other relevant abnormalities observed. | Both lungs show signs of involvement in the form of scattered areas of hazy opacity and scattered consolidation... which is consistent with COVID-19 pneumonia and has gotten worse **since February 2nd, 3rd, and 3rd**. There is no fluid around the lungs, lymph nodes, or other significant issues seen. | **LoRA: Safe, Lower Readability** (retained raw jargon & OCR artifact)<br>**Full FT: Hallucination** (simplified jargon, but confabulated dates *"February 2nd, 3rd, and 3rd"*) |

### Interpretation of Case Study

The audit exposes a critical clinical vulnerability of unconstrained fine-tuning. Although Full Fine-Tuning achieved higher ROUGE scores ($0.7317$ vs. $0.7066$), it produced a dangerous hallucination in **Case 2**, inserting *"correctly positioned"* for an endotracheal tube sitting in the right main bronchus (a serious clinical malposition). Updating all $248\text{M}$ parameters caused Full FT to over-index on common training templates describing properly placed lines. In contrast, LoRA's low-rank bottleneck acts as a structural regulariser, keeping generation grounded in the encoder's hidden states without fabricating clinical claims.

**Case 3** highlights the trade-off of this regularisation: **layperson readability versus factual safety**. Because LoRA updates only $0.36\%$ of the network, it acts conservatively on noisy or complex inputs—retaining the technical phrase *"ground-glass opacity"* and copying the raw de-identification artifact `"as of 2 2 3 2 3"` verbatim. Conversely, while Full Fine-Tuning translated *"ground-glass opacity"* into *"hazy opacity"*, it over-extrapolated on `"2 2 3 2 3"` and hallucinated a fabricated timeline (*"since February 2nd, 3rd, and 3rd"*). In a patient portal, LoRA's conservative literalism is far safer than Full FT's fluent confabulation.

## Recommended Model Deployment

For real-world clinical translation, we recommend deploying **FLAN-T5-base with custom LoRA ($r = 8$)** over the Full Fine-Tuning checkpoint. Capturing $96.6\%$ of Full FT's lexical overlap with $36\%$ less training VRAM is secondary to the primary clinical requirement: LoRA's low-rank bottleneck regularises against template memorisation and prevents ungrounded diagnostic hallucinations. Because generative models remain probabilistic, deployment should be wrapped in the following two-tier verification framework:

```mermaid
flowchart TD
    Source(["Technical Radiology Report\n(Clinical Source Narrative)"])

    subgraph Tier1["Tier 1: Parameter-Efficient Neural Generation"]
        LoRA["FLAN-T5-Base + Custom LoRA (r = 8, α = 32)\nDeterministic Beam Search (k = 2)"]
        Draft["Candidate Plain-Language Summary\n+ Token Log-Probabilities"]
        LoRA --> Draft
    end

    subgraph RefPipe["Deterministic Clinical Reference Extraction"]
        Parser["Biomedical Entity & Acuity Parser\n(UMLS / RadGraph)"]
        RefState["Extracted Source Entities (E_source)\n& High-Acuity Pathology Flags"]
        Parser --> RefState
    end

    Source --> LoRA
    Source --> Parser

    subgraph Tier2["Tier 2: Automated Clinical Safety Verification"]
        G1["1. Entity Grounding Check\nVerify E_summary ⊆ E_source\n(Zero Hallucinations)"]
        G2["2. Acute Omission Watchlist\nVerify 0 Missed Critical Findings\n(e.g. Malposition, Pneumothorax)"]
        G3["3. Generation Confidence Filter\nVerify Minimum Token Probability\nτ ≥ 0.85"]
    end

    Draft --> G1
    Draft --> G2
    Draft --> G3
    RefState -.->|"Ground-Truth Entities"| G1
    RefState -.->|"Required Acute Entities"| G2

    Gate{"All 3 Clinical Safety\nGates Passed?"}

    G1 -->|"Grounded"| Gate
    G2 -->|"Complete"| Gate
    G3 -->|"Confident"| Gate

    Pass(["Release Verified Summary\nto Patient Portal"])
    Fail(["Withhold Summary & Escalate\nto Radiologist Review Queue"])

    Gate ==>|"PASS: Verified Safe"| Pass
    Gate -.->|"FAIL: Safety Trigger"| Fail

    classDef inputNode fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0c4a6e
    classDef tier1Node fill:#ede9fe,stroke:#6d28d9,stroke-width:2px,color:#2e1065
    classDef parserNode fill:#ccfbf1,stroke:#0d9488,stroke-width:2px,color:#115e59
    classDef guardNode fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#78350f
    classDef gateNode fill:#f8fafc,stroke:#334155,stroke-width:2.5px,color:#0f172a
    classDef passNode fill:#dcfce7,stroke:#15803d,stroke-width:2.5px,color:#14532d
    classDef failNode fill:#fee2e2,stroke:#b91c1c,stroke-width:2.5px,color:#7f1d1d

    class Source inputNode
    class LoRA,Draft tier1Node
    class Parser,RefState parserNode
    class G1,G2,G3 guardNode
    class Gate gateNode
    class Pass passNode
    class Fail failNode

    style Tier1 fill:#f5f3ff,stroke:#8b5cf6,stroke-width:1.5px,stroke-dasharray: 5 5,color:#4c1d95
    style RefPipe fill:#f0fdfa,stroke:#14b8a6,stroke-width:1.5px,stroke-dasharray: 5 5,color:#134e4a
    style Tier2 fill:#fffbeb,stroke:#f59e0b,stroke-width:2px,color:#78350f
```

### Two-Tier Guardrail Architecture

Tier 1 generates candidate summaries using `google/flan-t5-base` with custom LoRA ($r=8, \alpha=32$) under deterministic beam search ($k=2$, length penalty $1.0$). Stochastic sampling is disabled entirely to guarantee consistent, repeatable clinical translations. Before reaching the patient portal, Tier 2 intercepts each draft through three automated checks: an entity grounding filter (RadGraph/UMLS) enforcing $E_{\text{summary}} \subseteq E_{\text{source}}$ to block hallucinations, an acute watchlist preventing omissions of critical pathologies (like pneumothoraces or tube malpositions), and a confidence gate requiring minimum token probabilities of $\tau \ge 0.85$. Summaries clearing all three gates release automatically; any flagged case routes directly to an attending radiologist for review.

### Operational & Hardware Sizing

Inference requires just $1.69\text{ GB}$ of VRAM, allowing deployment on budget cloud GPUs (NVIDIA T4) or standard consumer hardware (RTX 3060/4060) without multi-GPU overhead. Generating at $307.8\text{ tokens/second}$ yields an average latency under $0.25\text{ seconds}$ per report, easily supporting real-time hospital portal workloads.

## Dependencies & Execution Guide

All experiments were implemented using PyTorch modules, without any high-level library wrappers.

### Dependencies & Requirements

| Component | Minimum Version & Role |
| :--- | :--- |
| **Python / CUDA** | Python $\ge 3.10$ with CUDA 12.1 (`bfloat16` support) |
| **PyTorch (`torch`)** | $\ge 2.1.0$ — custom LoRA modules, loss masking, and DDP |
| **Hugging Face (`transformers`, `datasets`)** | $\ge 4.38.0$ / $\ge 2.18.0$ — `flan-t5-base` weights and `LaymanRRG` data |
| **Evaluation (`evaluate`, `rouge-score`)** | $\ge 0.4.1$ / $\ge 0.1.2$ — ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum metrics |
| **Utilities (`sentencepiece`, `matplotlib`)** | $\ge 0.2.0$ / $\ge 3.8.0$ — T5 tokeniser backend and loss plotting |

#### Automated Installation Script

Below is a script that creates a virtual environment and installs the dependencies needed to run the project.

```bash
conda create -n clinical-flan python=3.10 -y
conda activate clinical-flan
pip install torch transformers datasets evaluate rouge-score sentencepiece matplotlib
```

### Reproducibility & CLI Execution

All pipelines execute directly from `recognition/FLAN-T5-RRG-s4436813/` (pass `--num_samples 500` for rapid ~2-minute validation benchmarking as reported in the table above, or `--num_samples -1` to evaluate across the entire split):

| Pipeline Stage | CLI Command | Generated Artifacts & Telemetry |
| :--- | :--- | :--- |
| **1. Zero-Shot Eval** | `python predict.py --mode zero_shot --num_samples 500` | Baseline ROUGE (`Val_Full` & `Val_Unseen`), VRAM & throughput |
| **2. Train LoRA (1 GPU)** | `python train.py --mode lora` | Saves `checkpoints/lora.pt` and plots `lora_loss.png` |
| **3. Train LoRA (DDP)** | `torchrun --nproc_per_node=2 train.py --mode lora` | Multi-GPU Distributed Data Parallel training |
| **4. Train Full FT** | `python train.py --mode full_finetune` | Saves `checkpoints/full_finetune.pt` ($100\%$ weights) |
| **5. Evaluate LoRA** | `python predict.py --mode lora --checkpoint checkpoints/lora.pt --num_samples 500` | LoRA ROUGE scores, latency telemetry & clinical safety autopsy |
| **6. Evaluate Full FT** | `python predict.py --mode full_finetune --checkpoint checkpoints/full_finetune.pt --num_samples 500` | Full FT ROUGE scores & comparative safety autopsy |

## Artificial Intelligence Usage Disclosure

In compliance with COMP3710 Assessment Guidelines, Generative AI usage is documented below:
* **AI Tools Utilised:** Google Gemini (via the Antigravity development environment).
* **Specific Tasks Assisted:** Styling the Mermaid deployment flowchart, scaffolding the Matplotlib loss-plotting layout (`lora_loss.png`), drafting Slurm batch templates (`sbatch`) for Rangpur, creating markdown tables given data, and proofreading report formatting.
* **Auditing, Verification & Empirical Testing:** Manually derived and verified the LoRA update ($h = W_0 x + \frac{\alpha}{r} B A x$) and zero-initialisation ($B = 0$) to guarantee zero step-0 perturbation; audited `dataset.py` to confirm target padding (`0`) maps strictly to `-100` for `CrossEntropyLoss` and verified SHA-256 deduplication for `Val_Unseen`; empirically tested single-GPU and `torchrun` DDP execution on NVIDIA A100 nodes.
* **Verification Responsibility:** I assumed 100% personal responsibility for every line of code, loss formula, metric, and clinical interpretation in this submission.

## References

1. C. Raffel et al., "Exploring the limits of transfer learning with a unified text-to-text transformer," *Journal of Machine Learning Research*, vol. 21, no. 140, pp. 1–67, 2020.
2. H. W. Chung et al., "Scaling instruction-finetuned language models," *Journal of Machine Learning Research*, vol. 25, no. 70, pp. 1–53, 2024 (arXiv:2210.11416).
3. E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," in *Proc. International Conference on Learning Representations (ICLR)*, 2022.
4. BioLaySumm 2025 Shared Task Organisers, "BioLaySumm 2025 Subtask 2.1: Layman Radiology Report Generation (`LaymanRRG` Open-Source Track)," in *Proc. ACL 2025 Workshop on Biomedical Lay Summarisation*, 2025. Available: `https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track`
5. C.-Y. Lin, "ROUGE: A package for automatic evaluation of summaries," in *Text Summarization Branches Out (ACL Workshop)*, 2004, pp. 74–81.
6. S. Jain et al., "RadGraph: Extracting clinical entities and relations from radiology reports," in *Proc. NeurIPS Datasets and Benchmarks Track*, 2021.
