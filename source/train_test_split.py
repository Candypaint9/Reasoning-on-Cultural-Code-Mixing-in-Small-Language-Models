import pandas as pd
from sklearn.model_selection import train_test_split
import os

FILES_TO_SPLIT = [
    r"C:\Users\advai\Documents\Capstone\final_dataset\l2_codemixed_final.csv",
    r"C:\Users\advai\Documents\Capstone\final_dataset\l3_codemixed_final.csv"
]

TEST_SIZE = 0.10  


def get_anchor_region(region_str):
    if pd.isna(region_str):
        return region_str
    # Split by comma, clean up spaces, and grab the absolute last one
    parts = [r.strip() for r in str(region_str).split(',')]
    return parts[-1] if parts else region_str


def main():
    for file_path in FILES_TO_SPLIT:
        if not os.path.exists(file_path):
            print(f"\n[ERROR] Could not find {file_path}")
            continue
            
        filename = os.path.basename(file_path)
        prefix = filename.split('_')[0]  
        
        print(f"\n{'='*50}")
        print(f"Splitting: {filename}")
        print(f"{'='*50}")
        
        df = pd.read_csv(file_path)
        
        if 'question_region' in df.columns:
            initial_len = len(df)
            df = df.dropna(subset=['question_region'])
            if len(df) < initial_len:
                print(f"Dropped {initial_len - len(df)} rows missing region data.")

            # --- APPLY THE L3 HEURISTIC ---
            df['primary_region'] = df['question_region'].apply(get_anchor_region)

            counts = df['primary_region'].value_counts()
            singletons = counts[counts < 2].index
            
            df_singletons = df[df['primary_region'].isin(singletons)]
            df_stratifiable = df[~df['primary_region'].isin(singletons)]
            
            print(f"Using anchor regions. Found {len(df_singletons)} singletons to push to Training Set.")

            # Stratify based on the anchor region, not the full permutation
            train_strat, test_df = train_test_split(
                df_stratifiable, 
                test_size=TEST_SIZE, 
                stratify=df_stratifiable['primary_region'], 
                random_state=42 
            )
            
            train_df = pd.concat([train_strat, df_singletons])
            train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

            print("\nVerification (Top 3 Anchor Regions):")
            print("FULL Dataset Proportions:")
            print((df['primary_region'].value_counts(normalize=True) * 100).head(3).round(1).astype(str) + '%')
            
            print("\nTEST Set Proportions (Should closely match above):")
            print((test_df['primary_region'].value_counts(normalize=True) * 100).head(3).round(1).astype(str) + '%')
            
            train_df = train_df.drop(columns=['primary_region'])
            test_df = test_df.drop(columns=['primary_region'])
            
        else:
            print(f"[WARNING] 'question_region' column not found! Falling back to standard random split.")
            train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42)

        output_dir = os.path.dirname(file_path)
        train_path = os.path.join(output_dir, f"train_{prefix}.csv")
        test_path = os.path.join(output_dir, f"test_{prefix}.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"\n[SUCCESS] Split completed for {prefix.upper()}:")
        print(f" -> Training Data: {len(train_df)} rows saved to {train_path}")
        print(f" -> Holdout Data:  {len(test_df)} rows saved to {test_path}")

if __name__ == "__main__":
    main()