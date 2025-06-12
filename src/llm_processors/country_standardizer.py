
from src.llm_processors.base import LlmProcessor
import pandas as pd
from pathlib import Path
from typing import Union
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
        replacement_maps: Union[Path, dict] = None,
        frequent_unsupported: Union[Path, list] = None,
    ):
        """
        Initialize CountryNameStandardizer

        Args:
            df: input data  
            prompt: prompt to perform country name standardization
            std_country_set: Predefined set or file path of standard country names.
            replacement_maps: Dictionary or file path containing country name replacement mappings
            frequent_unsupported: List or file path of frequently seen unsupported country names.
        """
        self.df = df
        self.prompt = prompt
        self.std_country_set = read_standard_country(std_country_set)
        self.replacement_maps = read_dicts_from_yaml(replacement_maps)
        self.frequent_unsupported = read_yaml(frequent_unsupported)['frequent_unsupported'] if frequent_unsupported is not None else None
        
        self.valid_already_df = None
        self.to_process_df = self.df

    
    def preprocess(self, input_column: str = 'firm_export_country'):
        """
        Preprocess the input DataFrame to separate already standardized country names from those needing further processing.
       
        The method updates:
        - self.valid_already_df: DataFrame containing rows where no country name needs to be standardized.
        - self.to_process_df: DataFrame containing rows with at least one country name needing processing
        """
        def split_valid_toprocess(row):

            countries = self.preprocess_single(str(row[input_column]), 
                                                self.replacement_maps,
                                                self.frequent_unsupported)

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
    def preprocess_single(
        country_str : str,
        replacement_maps : Union[Path, None], 
        frequent_unsupported : Union[Path, None]
    ):
        """
        Preprocess a single country string by filtering and replacement. 

        Args:
            country_str (str): The input string representing one or more country names.
            replacement_maps (dict or None): A dictionary (or dict of dicts) containing replacement mappings for country names.
            frequent_unsupported (list or None): A list of country names to be excluded from processing.
        
        Returns:
            list: A list of processed country names after filtering and replacement.
        """

        if pd.isna(country_str):
            return country_str

        countries = read_list_string_from_df(country_str)

        if frequent_unsupported:
            countries = [c for c in countries if c not in frequent_unsupported]

        if replacement_maps:
            for re_map in replacement_maps.values():
                countries = [re_map.get(c, c) for c in countries]
    
        return countries




