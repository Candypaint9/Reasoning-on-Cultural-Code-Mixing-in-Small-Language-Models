import pandas as pd
import ast

input_file = "codemixed_dataset.csv" 
output_file = "final_codemixed_dataset.csv"

df = pd.read_csv(input_file)
print(f"Loaded {len(df)} rows.")
print(f"Current columns safely detected: {df.columns.tolist()}")

def parse_options(opt_string):
    try:
        parsed_list = ast.literal_eval(str(opt_string))
        if isinstance(parsed_list, list):
            while len(parsed_list) < 4:
                parsed_list.append("")
            return parsed_list[:4] 
    except (ValueError, SyntaxError):
        pass
    return ["", "", "", ""] 

parsed_options = df['options'].apply(parse_options)
parsed_df = pd.DataFrame(parsed_options.tolist(), columns=['option1', 'option2', 'option3', 'option4'], index=df.index)

options_idx = df.columns.get_loc('options')

df = df.drop(columns=['options'])

df.insert(options_idx, 'option1', parsed_df['option1'])
df.insert(options_idx + 1, 'option2', parsed_df['option2'])
df.insert(options_idx + 2, 'option3', parsed_df['option3'])
df.insert(options_idx + 3, 'option4', parsed_df['option4'])

df.to_csv(output_file, index=False, encoding='utf-8')

print(f"Final columns preserved: {df.columns.tolist()}")
print(f"Saved to: {output_file}")