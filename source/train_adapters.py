import os
import pathlib

# --- THE WINDOWS UTF-8 PATCH ---
_original_read_text = pathlib.Path.read_text
def _patched_read_text(self, encoding=None, errors=None):
    if encoding is None:
        encoding = 'utf-8'
    return _original_read_text(self, encoding=encoding, errors=errors)
pathlib.Path.read_text = _patched_read_text
# --------------------------------------

import random
import torch
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "./llama-1b-cultural-adapter-local-sampled"
TRAIN_FILES = [
    r"C:\Users\advai\Documents\Capstone\final_dataset\train_l2.csv", 
    r"C:\Users\advai\Documents\Capstone\final_dataset\train_l3.csv"
]

SAMPLE_PERCENTAGE = 1

# DATA PREPARATION
def extract_all_prompts(row):
    options = [str(row['option1']), str(row['option2']), str(row['option3']), str(row['option4'])]
    answer_text = str(row['answer'])
    
    try:
        correct_idx = options.index(answer_text)
        correct_letter = ['A', 'B', 'C', 'D'][correct_idx]
    except ValueError:
        return [] 
    
    base_template = "Question: {q}\nOptions:\nA. {o[0]}\nB. {o[1]}\nC. {o[2]}\nD. {o[3]}\nAnswer: {ans}"
    prompts = []
    
    if 'question' in row and pd.notna(row['question']):
        prompts.append({"text": base_template.format(q=row['question'], o=options, ans=correct_letter)})
    if 'question_codemixed' in row and pd.notna(row['question_codemixed']):
        prompts.append({"text": base_template.format(q=row['question_codemixed'], o=options, ans=correct_letter)})
    if 'codemixed_q1' in row and pd.notna(row['codemixed_q1']):
        prompts.append({"text": base_template.format(q=row['codemixed_q1'], o=options, ans=correct_letter)})
    if 'codemixed_q2' in row and pd.notna(row['codemixed_q2']):
        prompts.append({"text": base_template.format(q=row['codemixed_q2'], o=options, ans=correct_letter)})
        
    return prompts

print(f"Loading datasets and sampling {SAMPLE_PERCENTAGE * 100}% of the variations...")
all_formatted_data = []
for file in TRAIN_FILES:
    if os.path.exists(file):
        df = pd.read_csv(file)
        df = df.dropna(subset=['option1', 'answer']) 
        for _, row in df.iterrows():
            all_formatted_data.extend(extract_all_prompts(row))
    else:
        print(f"[WARNING] Could not find {file} locally.")

random.shuffle(all_formatted_data)

sample_size = int(len(all_formatted_data) * SAMPLE_PERCENTAGE)
sampled_data = all_formatted_data[:sample_size]

train_dataset = Dataset.from_list(sampled_data)
print(f"Total balanced training samples: {len(train_dataset)}")



print("Loading model in pure 16-bit with SDPA acceleration...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",  
    token=HF_TOKEN
)


peft_config = LoraConfig(
    r=16,               
    lora_alpha=16,     
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# TRAINING LOOP WITH COSINE WARMUP
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,  
    gradient_accumulation_steps=2,  
    learning_rate=2e-4,
    lr_scheduler_type="cosine",     
    warmup_ratio=0.1,               
    logging_steps=50,
    num_train_epochs=1,             
    save_steps=500,
    optim="adamw_torch",            
    bf16=True,            
    report_to="none",     
    dataset_text_field="text", 
    max_length=256        
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,   
    args=training_args,
)

print("Starting global sampled adapter training...")
trainer.train()

trainer.model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print(f"\n[SUCCESS] Local global adapter saved to {OUTPUT_DIR}/final")