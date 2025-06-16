import os
import pandas as pd
from tqdm import tqdm 
from models.vllm_openai_compat_model import VLLMOpenAICompatModel


def task_run(df : pd.DataFrame, 
             llm : VLLMOpenAICompatModel, 
             prompt :str, 
             input_column : str, 
             new_column : str, 
             temp_path : str):
    
    """Process dataframe with LLM, saving progress incrementally"""
    
    # Load or initialize temporary results
    if os.path.exists(temp_path):
        temp_df = pd.read_csv(temp_path)
        processed_indices = set(temp_df['index'])
    else:
        temp_df = pd.DataFrame(columns=['index', new_column])
        processed_indices = set()
    
    # Process rows with progress tracking
    for index, row in tqdm(df.iterrows(), total=len(df)):

        if index in processed_indices:
            continue  
            
        try:
            result = llm.get_llm_response(prompt, str(row[input_column]))
            
            # Store result in both DataFrames
            df.at[index, new_column] = result
            temp_df = pd.concat([
                temp_df,
                pd.DataFrame([{'index': index, new_column: result}])
            ])

            temp_df.to_csv(temp_path, index=False)
            
        except Exception as e:
            print(f"Skipping row {index} due to error: {str(e)}")
            continue

    return df, temp_path
