
from src.llm_processors.base import LlmProcessor
import pandas as pd
from pathlib import Path
from typing import Union, Set
import logging

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from utils.country_name_io import *



class CountryNameStandardizer(LlmProcessor):
    
    def __init__(
        self,
        df: pd.DataFrame,
        prompt: str,
        std_country_set: Union[Path, set],
        country_code_mapping: Union[Path, dict] = None
    ):
        """
        Initialize CountryNameStandardizer

        Args:
            df: input data  
            prompt: prompt to perform country name standardization
            std_country_set: predefined standard country names
            country_code_mapping: 
        """
        self.df = df
        self.prompt = prompt
        self.std_country_set = read_standard_country(std_country_set)
        self.country_code_mapping = read_country_code_mapping(country_code_mapping)

        self.valid_already_df = None
        self.to_process_df = self.df

    
    def preprocess(self, input_column: str = 'firm_export_country'):
        
        def split_valid_toprocess(row):

            countries = self.pre_process_single(str(row[input_column]), self.country_code_mapping)

            valid_country_name = [c for c in countries if c in self.std_country_set]
            country_name_to_process= [c for c in countries if c and c not in self.std_country_set]

            return pd.Series([valid_country_name, country_name_to_process])
        
        self.df[['valid_already', 'to_process']] = self.df.apply(split_valid_toprocess, axis=1)
        
        self.valid_already_df = self.df[self.df['to_process'].str.len() == 0]
        self.to_process_df = self.df[self.df['to_process'].str.len() > 0]
    
        logger.info(f'Country name standardization: {len(self.to_process_df)} out of {len(self.df)} data need processing')

    
    def llm_process(self):
        # run one batch
        # return result list
        pass

    def postprocess(self):
        pass


    @staticmethod
    def pre_process_single(country_str, country_code_mapping):

        if pd.isna(country_str):
            return country_str

        countries = read_list_string_from_df(country_str)
        #countries = [c for c in countries if c not in frequent_unsupported]

        # Replace names
        if country_code_mapping:
            countries = [country_code_mapping.get(c, c) for c in countries]

        #countries = [replace_map.get(c, c) for c in countries]

        return countries



    


