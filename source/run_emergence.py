import os
import torch
import pandas as pd
import numpy as np
import gc
import json
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found! Please set it using 'export HF_TOKEN=...'")

MODELS = [
    # {
    #     "path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    #     "adapter": None,
    #     "name": "Meta-Llama-3.1-8B-Instruct"
    # },
    # {
    #     "path": "meta-llama/Llama-3.2-3B-Instruct",
    #     "adapter": None,
    #     "name": "Llama-3.2-3B-Instruct"
    # },
    # {
    #     "path": "meta-llama/Llama-3.2-1B-Instruct",
    #     "adapter": None,
    #     "name": "Llama-3.2-1B-Instruct"
    # },
    {
        "path": "meta-llama/Llama-3.2-1B-Instruct",
        "adapter": "./llama-1b-ft-adapter",
        "name": "Llama-3.2-1B-Instruct-FT"
    }
]

DATASETS = [
    {"file": "dataset/final_dataset/sanskriti_dataset.csv", "qtype": "L1", "samples": 2000},
    {"file": "dataset/final_dataset/test_l2.csv",           "qtype": "L2", "samples": -1},
    {"file": "dataset/final_dataset/test_l3.csv",           "qtype": "L3", "samples": -1}
]

THRESHOLDS = [0.0, 0.25, 0.5, 1.0, 2.0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OPTION_LETTERS = ['A', 'B', 'C', 'D']

def get_final_norm(model_instance):
    if hasattr(model_instance, "base_model"):
        model_instance = model_instance.base_model

    base_model = getattr(model_instance, "model", model_instance)
    if hasattr(base_model, "norm"): return base_model.norm
    if hasattr(base_model, "final_layernorm"): return base_model.final_layernorm
    if hasattr(base_model, "layer_norm"): return base_model.layer_norm
    for name, module in reversed(list(base_model.named_modules())):
        if "norm" in module.__class__.__name__.lower():
            return module
    raise AttributeError("Could not find the final layer norm!")

def calculate_emergence_multi(layer_logits_dict, correct_letter, thresholds):
    incorrect_letters = [l for l in OPTION_LETTERS if l != correct_letter]
    num_layers = len(layer_logits_dict[correct_letter])
    results = {t: np.nan for t in thresholds}
    thresholds_to_find = list(thresholds)
    delta_list = []

    final_correct_logit = layer_logits_dict[correct_letter][-1]
    final_max_incorrect = max(layer_logits_dict[l][-1] for l in incorrect_letters)
    is_correct = 1 if final_correct_logit > final_max_incorrect else 0

    for layer_idx in range(num_layers):
        correct_logit = layer_logits_dict[correct_letter][layer_idx]
        max_incorrect_logit = max(layer_logits_dict[l][layer_idx] for l in incorrect_letters)
        delta = correct_logit - max_incorrect_logit
        delta_list.append(delta)

        for t in thresholds_to_find[:]:
            if delta > t:
                results[t] = layer_idx + 1
                thresholds_to_find.remove(t)
    return results, delta_list, is_correct

def calculate_stats(emergence_list):
    valid_layers = [x for x in emergence_list if not np.isnan(x)]
    if not valid_layers: return [None] * 7
    return [
        round(np.mean(valid_layers), 2), round(np.median(valid_layers), 2), round(np.std(valid_layers), 2),
        np.min(valid_layers), np.percentile(valid_layers, 25), np.percentile(valid_layers, 75), np.max(valid_layers)
    ]

def clean_for_json(obj):
    if isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

def load_and_sample_dataset(file_path, num_samples):
    print(f"\nReading dataset from {file_path}...")
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}.")
        return None

    df_full = pd.read_csv(file_path)

    if num_samples != -1 and len(df_full) > num_samples:
        print(f"Sampling {num_samples} rows while balancing regions...")
        if 'question_region' in df_full.columns:
            df_full = df_full.sample(frac=1, random_state=42)
            region_counts = df_full['question_region'].value_counts()
            regions = list(region_counts.index[::-1])

            keep_indices = []
            remaining_needed = num_samples

            for i, region in enumerate(regions):
                take = min(region_counts[region], remaining_needed // (len(regions) - i))
                keep_indices.extend(df_full[df_full['question_region'] == region].head(take).index)
                remaining_needed -= take

            df = df_full.loc[keep_indices].sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"Sampled region distribution (Top 5):\n{df['question_region'].value_counts().head(5)}")
        else:
            print("Warning: 'question_region' column not found. Doing a standard random sample.")
            df = df_full.sample(n=num_samples, random_state=42).reset_index(drop=True)
    else:
        print(f"Using all {len(df_full)} rows from the dataset.")
        df = df_full

    return df


def main():
    for model_info in MODELS:
        model_path = model_info["path"]
        adapter_path = model_info.get("adapter")
        short_name = model_info["name"]

        print(f"\n{'='*60}\nLoading Model into VRAM: {short_name}\n{'='*60}")

        use_4bit = "70b" in short_name.lower()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        TOKEN_IDS = {letter: tokenizer.encode(letter, add_special_tokens=False)[-1] for letter in OPTION_LETTERS}

        model_kwargs = {"device_map": {"": 0}, "output_hidden_states": True}

        if use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
            )
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        # 1. Load Base Model
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, **model_kwargs)

        # 2. Attach Adapter if specified
        if adapter_path:
            print(f"Attaching LoRA Adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)

        model.eval()

        # 3. Loop through all datasets with the loaded model
        for ds in DATASETS:
            qtype = ds["qtype"]
            df = load_and_sample_dataset(ds["file"], ds["samples"])

            if df is None:
                continue

            has_q_codemixed = 'question_codemixed' in df.columns
            has_q1 = 'codemixed_q1' in df.columns
            has_q2 = 'codemixed_q2' in df.columns

            emergence_std = {t: [] for t in THRESHOLDS}
            emergence_cm_combined = {t: [] for t in THRESHOLDS}
            std_deltas_accum, cm_deltas_accum = [], []
            std_correct, std_total, cm_correct, cm_total = 0, 0, 0, 0

            for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Inferencing {short_name} on {qtype}"):
                options = [str(row['option1']), str(row['option2']), str(row['option3']), str(row['option4'])]
                answer_text = str(row['answer'])
                try: correct_letter = OPTION_LETTERS[options.index(answer_text)]
                except ValueError: continue

                def process_query(q_text, is_cm=False):
                    if pd.isna(q_text) or str(q_text).strip() == "": return
                    prompt = f"Question: {q_text}\nOptions:\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nAnswer:"
                    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

                    with torch.no_grad(): outputs = model(**inputs, output_hidden_states=True)

                    layer_logits_dict = {l: [] for l in OPTION_LETTERS}
                    final_norm = get_final_norm(model)
                    for l in range(1, len(outputs.hidden_states)):
                        logits = model.lm_head(final_norm(outputs.hidden_states[l][0, -1, :]))
                        for letter in OPTION_LETTERS: layer_logits_dict[letter].append(logits[TOKEN_IDS[letter]].item())

                    res, deltas, is_correct = calculate_emergence_multi(layer_logits_dict, correct_letter, THRESHOLDS)

                    if is_cm:
                        cm_deltas_accum.append(deltas)
                        for t in THRESHOLDS:
                            if not np.isnan(res[t]): emergence_cm_combined[t].append(res[t])
                        nonlocal cm_correct, cm_total
                        cm_correct += is_correct
                        cm_total += 1
                    else:
                        std_deltas_accum.append(deltas)
                        for t in THRESHOLDS: emergence_std[t].append(res[t])
                        nonlocal std_correct, std_total
                        std_correct += is_correct
                        std_total += 1

                if 'question' in df.columns: process_query(row['question'], is_cm=False)
                if has_q_codemixed: process_query(row['question_codemixed'], is_cm=True)
                if has_q1: process_query(row['codemixed_q1'], is_cm=True)
                if has_q2: process_query(row['codemixed_q2'], is_cm=True)

            # ------------------------------------------
            # APPEND DATA TO THE SPECIFIC MASTER JSON
            # ------------------------------------------
            model_results = {
                'std_means': [calculate_stats(emergence_std[t])[0] for t in THRESHOLDS],
                'cm_means': [calculate_stats(emergence_cm_combined[t])[0] for t in THRESHOLDS],
                'avg_std_deltas': np.mean(std_deltas_accum, axis=0).tolist() if std_deltas_accum else [],
                'avg_cm_deltas': np.mean(cm_deltas_accum, axis=0).tolist() if cm_deltas_accum else [],
                'std_acc': (std_correct / std_total * 100) if std_total > 0 else 0,
                'cm_acc': (cm_correct / cm_total * 100) if cm_total > 0 else 0
            }

            master_json_file = f"{qtype}_master_results.json"

            if os.path.exists(master_json_file):
                with open(master_json_file, "r") as f:
                    all_results = json.load(f)
            else:
                all_results = {}

            all_results[short_name] = clean_for_json(model_results)

            with open(master_json_file, "w") as f:
                json.dump(all_results, f, indent=4)

            print(f" -> [SUCCESS] Appended {short_name} data to {master_json_file}")

        print(f"\nCleaning VRAM for {short_name}...")
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()