import pandas as pd
import requests
import time
import re
import os
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()


ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
API_TOKEN  = os.getenv("CLOUDFLARE_API_TOKEN")

MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"

HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

INPUT_CSV = "fixed_sorted_by_question_region.csv"
ENTITIES_CSV = "entities.csv"
OUTPUT_CSV = "cloudflare_codemixed_dataset.csv"


SUPPORTED_LANGUAGES = {
    "Hindi", "Bengali", "Marathi", "Telugu", "Tamil", "Gujarati", 
    "Urdu", "Kannada", "Odia", "Malayalam", "Punjabi", "Assamese", "Kashmiri"
}

region_lang_map = {
    "Andaman_And_Nicobar": "Bengali", "Andhra_Pradesh": "Telugu",      
    "Assam": "Assamese", "Bihar": "Hindi", "Chandigarh": "Hindi",          
    "Chhattisgarh": "Hindi", "Dadra_And_Nagar_Haveli_And_Daman_And_Diu": "Gujarati", 
    "Delhi": "Hindi", "Goa": "Konkani", "Gujarat": "Gujarati",             
    "Haryana": "Hindi", "Himachal_Pradesh": "Hindi", "Jammu_Kashmir": "Kashmiri",       
    "Ladakh": "Kashmiri", "Jharkhand": "Hindi", "Karnataka": "Kannada",           
    "Kerala": "Malayalam", "Lakshadweep": "Malayalam", "Madhya_Pradesh": "Hindi",      
    "Maharashtra": "Marathi", "Manipur": "Meitei", "Mizoram": "Mizo",             
    "Odisha": "Odia", "Puducherry": "Tamil", "Punjab": "Punjabi",              
    "Rajasthan": "Hindi", "Sikkim": "Nepali", "Tamil_Nadu": "Tamil",          
    "Telangana": "Telugu", "Tripura": "Bengali", "Uttar_Pradesh": "Hindi",       
    "Uttarakhand": "Hindi", "West_Bengal": "Bengali"          
}

def get_target_language(region):
    primary_lang = region_lang_map.get(region, "Hindi")
    if primary_lang not in SUPPORTED_LANGUAGES:
        return "Hindi" 
    return primary_lang



def generate_codemix_cf(question, target_lang, all_entities):
    entities_in_question = [ent for ent in all_entities if re.search(r'\b' + re.escape(ent) + r'\b', question, re.IGNORECASE)]
    
    entity_instruction = ""
    if entities_in_question:
        entity_list = "\n".join([f"- {e}" for e in entities_in_question])
        entity_instruction = f"\nCultural entities (keep exactly as-is in English, do NOT translate):\n{entity_list}\n"

    user_prompt = f"""You are a fluent bilingual speaker of English and {target_lang}. You naturally mix {target_lang} and English the way people really talk in casual conversations.

Original English question:
{question}
{entity_instruction}
Task:
Rewrite ONLY the QUESTION in a natural {target_lang}-English codemixed style.
Rules:
- Mix English + {target_lang} words/phrases very naturally (like daily conversation).
- Use the native script of {target_lang} for the {target_lang} words, and English script for the English words.
- Keep the exact same meaning, logic, and reasoning level.
- Make it sound informal, fluent, and authentic.
- NEVER output the options, answer, or any explanation.
- Output ONLY the rewritten codemixed question — nothing else.

Codemixed question:"""

    payload = {
        "messages": [
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.6,
        "top_p": 0.92
    }

    try:
        resp = requests.post(
            f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{MODEL}",
            headers=HEADERS,
            json=payload,
            timeout=50
        )
        resp.raise_for_status()
        result = resp.json()
        new_q = result.get('result', {}).get('response', '').strip()
        
        if new_q.startswith('"') and new_q.endswith('"'):
            new_q = new_q[1:-1]
            
        return new_q if new_q else "[EMPTY RESPONSE]"

    except requests.exceptions.RequestException as e:
        print(f"API Error: {str(e)}")
        return ""
    except Exception as e:
        print(f"Exception: {str(e)}")
        return ""


if __name__ == "__main__":
    try:
        df = pd.read_csv(OUTPUT_CSV)
        print(f"Resume file '{OUTPUT_CSV}' found.")
    except FileNotFoundError:
        try:
            df = pd.read_csv(INPUT_CSV)
            print(f"Starting fresh from '{INPUT_CSV}'...")
        except FileNotFoundError:
            print(f"Error: '{INPUT_CSV}' not found.")
            exit()

    try:
        entities_df = pd.read_csv(ENTITIES_CSV)
    except FileNotFoundError:
        print(f"Error: '{ENTITIES_CSV}' not found.")
        exit()

    all_entities = entities_df['entity'].dropna().astype(str).tolist()
    all_entities.sort(key=len, reverse=True)

    for col in ["lang1", "codemixed_q1", "lang2", "codemixed_q2"]:
        if col not in df.columns:
            df[col] = ""

    print(f"Loaded {len(df)} rows.")

    try:
        for i in tqdm(df.index, desc="Codemixing"):
            if pd.notna(df.loc[i, "codemixed_q1"]) and str(df.loc[i, "codemixed_q1"]).strip() != "":
                continue

            question = str(df.loc[i, "question"])
            regions_str = str(df.loc[i, "question_region"])
            
            if pd.isna(regions_str) or regions_str == "nan" or regions_str.strip() == "":
                continue
                
            regions = [r.strip() for r in regions_str.split(",")]
            
            region1 = regions[0]
            lang1 = get_target_language(region1)
            
            mixed_q1 = generate_codemix_cf(question, lang1, all_entities)
            if mixed_q1:
                df.loc[i, "lang1"] = lang1
                df.loc[i, "codemixed_q1"] = mixed_q1
                time.sleep(2.2) 
                
            if len(regions) > 1:
                region2 = regions[1]
                lang2 = get_target_language(region2)
                
                mixed_q2 = generate_codemix_cf(question, lang2, all_entities)
                if mixed_q2:
                    df.loc[i, "lang2"] = lang2
                    df.loc[i, "codemixed_q2"] = mixed_q2
                    time.sleep(2.2) 

            if i > 0 and i % 50 == 0:
                df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    except KeyboardInterrupt:
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        print(f"Progress saved to {OUTPUT_CSV}.")
        exit()

    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"\n Processed dataset saved to: {OUTPUT_CSV}")