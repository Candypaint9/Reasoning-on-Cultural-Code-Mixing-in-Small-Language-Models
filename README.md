# Emergence of Code-Mixed Relational Reasoning in Small Language Models

## Project Overview
This repository investigates how Small Language Models (SLMs) process complex multi-entity and comparative reasoning, specifically when queried in code-mixed languages (e.g., English-Hindi). 

We utilize the **Sanskriti Dataset** to evaluate whether SLMs rely purely on factual memorization or if they can execute structural reasoning. By tracking layer-wise confidence trajectories (emergence), we identify cognitive bottlenecks in base models and demonstrate how targeted LoRA adapters can mechanically inject structural routing without causing catastrophic forgetting.

## Dataset Formulation
To isolate reasoning from factual recall, we extracted cultural entities from the Sanskriti dataset and structurally transformed the base recall questions into three hierarchical cognitive levels:

* **L1: Factual Recall** (Direct retrieval from the dataset)
  * *Example:* "Where is the Outrigger canoe designs famous?" 
  * *Answer:* Nicobar district
* **L2: Multi-Entity Relational Reasoning** (Finding intersections)
  * *Example:* "Which region of Andaman_and_Nicobar is associated with both Hodi Craft and Outrigger canoe designs?"
  * *Correct:* Nicobar district
* **L3: Comparative & Elimination Reasoning** (Multi-hop logic)
  * *Example (Comparative):* "While Jarawa body painting is associated with South and Middle Andaman Islands, Hodi Craft is associated with which region?"
  * *Example (Elimination):* "Which of the following art forms is NOT associated with Nicobar district? (A) Hodi Craft, (B) Outrigger canoe designs, (C) Jarawa body painting."
  * *Correct:* Jarawa body painting

## Methodology & Pipeline
The project pipeline is divided into dataset mapping, model training, and mechanistic evaluation. The scripts should be executed in the following order:

1. `sanskriti_entity_mapping.py`: Extracts key cultural entities from the raw dataset to generate an entity map.
2. `codemix-cloudflare.py`: Generates the code-mixed variations of the L1, L2, and L3 questions.
3. `train_test_split.py`: Performs proportional, region-stratified splitting to ensure no data leakage between training and evaluation phases.
4. `train_adapters.py`: Trains the global LoRA adapters on the Llama-3.2-1B-Instruct model using targeted relational fine-tuning.
5. `run_emergence.py`: Inferences the test sets across the base and fine-tuned models, tracking the layer-wise logit differentials ($\Delta \ell$) to measure confidence growth.
6. `plot_models_compare.py` / `plot_dataset_compare.py`: Generates the final visualizations and comparative emergence trajectories.

## Results & Analysis
The following table summarizes the final layer accuracy across the L1, L2, and L3 datasets for standard English and Code-Mixed (CM) variations.

| Model | L1 (Std) | L2 (Std) | L2 (CM) | L3 (Std) | L3 (CM) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Meta-Llama-3.1-8B-Instruct** | 83.8% | 87.4% | 83.6% | 68.6% | 56.0% |
| **Llama-3.2-3B-Instruct** | 77.4% | 80.4% | 74.3% | 61.4% | 49.3% |
| **Llama-3.2-1B-Instruct (Base)** | 59.5% | 56.0% | 52.1% | 33.3% | 28.6% |
| **Llama-3.2-1B-Instruct-FT** (Full pipeline) | 79.2% | 99.4% | 99.1% | 92.5% | 90.7% |
| **Llama-3.2-1B-Instruct-FT-only-L1** (Ablation)| 92.0% | 81.9% | 81.3% | 46.1% | 33.7% |

### Key Findings
1. **The Reasoning Bottleneck:** Base models show significant degradation when moving from L1 (Recall) to L3 (Comparative), especially under code-mixed constraints. 
2. **Success of Structural Fine-Tuning:** The `1B-Instruct-FT` model effectively overcomes this bottleneck, achieving >90% accuracy on L3 Code-Mixed tasks, outperforming the much larger 8B baseline.
3. **Knowledge vs. Reasoning (Ablation):** The `FT-only-L1` ablation study proves that injecting factual knowledge alone achieves high recall (92%) but fails catastrophically on comparative logic (33.7%). This confirms the full adapter successfully learned structural reasoning, not just vocabulary memorization.

![L3 Accuracy Bars](results/plots/L3_unified_accuracy_bars.png)
![Confidence Growth Proof](results/plots/Llama-3.2-1B-Instruct-FT_cross_dataset_confidence_growth.png)
![Ablation Study](results/plots/Llama-3.2-1B-Instruct-FT-only-L1_cross_dataset_confidence_growth.png)
