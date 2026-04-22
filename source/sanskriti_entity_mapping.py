import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading {model_id} in 4-bit precision...")

tokenizer = AutoTokenizer.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config
)
print("Model loaded successfully onto the GPU!")

from datasets import load_dataset
import pandas as pd

print("Loading Sanskriti dataset from Hugging Face...")
dataset = load_dataset("13ari/Sanskriti", split="train")
df = dataset.to_pandas()

df = df.iloc[10001:]

def create_prompt(row):
    ans_val = str(row['answer']).strip()
    correct_option = ""
    mapping = {'1': 'option1', '2': 'option2', '3': 'option3', '4': 'option4',
               'A': 'option1', 'B': 'option2', 'C': 'option3', 'D': 'option4'}

    if ans_val.lower() in ['option1', 'option2', 'option3', 'option4']:
        correct_option = str(row.get(ans_val.lower(), ''))
    elif ans_val.upper() in mapping:
        correct_option = str(row.get(mapping[ans_val.upper()], ''))
    else:
        correct_option = ans_val

    state = str(row.get('state', '')).strip()
    category = str(row.get('attribute', '')).strip()
    question = str(row.get('question', '')).strip()

    target_text = f"{question} The correct answer is: {correct_option}."

    system_prompt = "You are a deterministic data extraction algorithm. You extract exact substrings from the provided text. You never hallucinate, guess, or use external knowledge."

    user_prompt = f"""
Analyze the following text about Indian culture to extract a specific entity.

Target Text: "{target_text}"

Context:
- Region: {state}
- Category: {category}

Task:
Extract the exact name of the {category} from the Target Text. Because the category is '{category}', you are looking for the specific proper noun associated with it (for example: a specific dance form, a local dish/food, an art style, a spoken language, a festival, a tribe, or a monument).

STRICT RULES (CRITICAL):
1. EXACT SUBSTRING: The extracted entity MUST be an exact, word-for-word match of a phrase found inside the Target Text. Do not alter spelling.
2. NO REGIONS: Do NOT output the name of the state ("{state}") or "India". We already know the region. We only want the cultural item.
3. NO BOILERPLATE: Ignore filler words like "Which of the following", "is a", or "belongs to". Extract ONLY the core proper noun.
4. JSON ONLY: Output ONLY a valid JSON object in this exact format: {{"entity": "Extracted Name"}}
5. Do not include markdown formatting, backticks, or any other conversational text.
"""
    return system_prompt, user_prompt

print("Dataset and prompt logic ready.")

import json
from tqdm import tqdm

success_path = '/content/llm_sanskriti_mapping.csv'
error_path = '/content/llm_errors.csv'

results = []
errors = []

print(f"Starting extraction for {len(df)} rows...")

for index, row in tqdm(df.iterrows(), total=len(df)):
    system_prompt, user_prompt = create_prompt(row)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=20, 
        temperature=0.1,  
        do_sample=True
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    try:
        # Find the first '{' and the last '}' to ignore any outside quotes or text
        start_idx = response.find('{')
        end_idx = response.rfind('}')

        if start_idx != -1 and end_idx != -1:
            # Extract only the JSON portion
            clean_response = response[start_idx:end_idx+1]
        else:
            clean_response = response

        extracted_data = json.loads(clean_response)

        results.append({
            'entity': extracted_data.get('entity', ''),
            'region': row.get('state', ''),
            'category': row.get('attribute', '')
        })
    except json.JSONDecodeError:
        row_dict = row.to_dict()
        row_dict['llm_raw_output'] = response
        errors.append(row_dict)

final_df = pd.DataFrame(results).dropna().drop_duplicates()
final_df.to_csv(success_path, index=False)
print(f"\nSuccessfully extracted {len(final_df)} entities!")
print(f"File saved to: {success_path} (Check the folder icon on the left toolbar)")

if errors:
    pd.DataFrame(errors).to_csv(error_path, index=False)
    print(f"Logged {len(errors)} formatting errors to: {error_path} (Check the folder icon)")
else:
    print("Zero errors encountered!")

from google.colab import files

success_file = '/content/llm_sanskriti_mapping.csv'
error_file = '/content/llm_errors.csv'

print(f"Downloading {success_file}...")
files.download(success_file)

print(f"Downloading {error_file} (if it exists)...")
files.download(error_file)

print("Download requests initiated. Please check your browser's download manager.")

