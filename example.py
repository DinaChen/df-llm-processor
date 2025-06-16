import pandas as pd
import yaml
from utils import *
from src import *
from pathlib import Path
from utils.country_name_io import *
from models.vllm_openai_compat_model import VLLMOpenAICompatModel
import pandas as pd

""" 
Usecase Example: 

"""

# TODO: parse input config from yaml and do validation


# countryname _standardizer config
data_path = ''
std_country = ''
in_column = 'firm_export_country'
out_column = 'llm_cleaned_country'
frequent_replacement = None
frequent_unsupported = None

# model config
url = "http://localhost:1234/v1"
model_name = "Qwen2.5-14B-Instruct"
prompt = """
prompt prompt prompt 
"""


countryt_name_processor = CountryNameStandardizer(
                                    Path(data_path), 
                                    prompt, 
                                    Path(std_country), 
                                    in_column,
                                    out_column,
                                    Path(frequent_replacement), 
                                    Path(frequent_unsupported)
                                    )

llm = VLLMOpenAICompatModel(None, url, model_name)

countryt_name_processor.preprocess()
countryt_name_processor.llm_process(llm)
countryt_name_processor.postprocess()

save_path = ''
countryt_name_processor.to_csv(save_path)
