
from src.llm_task_processors.base import LlmTaskProcessor
from utils.task_runner import task_run
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


class CountryNameStandardizer(LlmTaskProcessor):
    
    def __init__(
        self,
        df: Union[Path, pd.DataFrame],
        prompt: str,
        std_country_set: Union[Path, set],
        input_column: str,
        output_column:str = None,
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
        task_name = f'Country_std_data{len(df)}' if isinstance(df, pd.DataFrame) else f'Country_std_{df.stem}'

        self.output_dir = Path("outputs") / task_name  
        self.output_dir.mkdir(parents=True, exist_ok=True)  

        self.df = df if isinstance(df, pd.DataFrame) else pd.read_csv(df, index_col=False)
        self.std_country_set = read_standard_country(std_country_set)

        self.input_column = input_column
        self.output_column = output_column if output_column else task_name + '_result'

        self.prompt = prompt.format(std_country_string = str(self.std_country_set))
        self.replacement_maps = read_dicts_from_yaml(replacement_maps)
        self.frequent_unsupported = read_yaml(frequent_unsupported)['frequent_unsupported'] if frequent_unsupported is not None else None
        
        self.valid_already_df = None
        self.to_process_df = self.df

        self.to_remove=[]

        self.result_column_list = list(self.df.columns) + [self.output_column]
    
    def preprocess(self) -> None:
        """
        Preprocess the input DataFrame to separate already standardized country names from those needing further processing.
       
        The method updates:
        - self.valid_already_df: DataFrame containing rows where no country name needs to be standardized.
        - self.to_process_df: DataFrame containing rows with at least one country name needing processing
        """

        def split_valid_toprocess(row):


            countries = self.preprocess_single(str(row[self.input_column]), 
                                                self.replacement_maps,
                                                self.frequent_unsupported)

            valid_country_name = [c for c in countries if c in self.std_country_set]
            country_name_to_process= [c for c in countries if c and c not in self.std_country_set]

            return pd.Series([valid_country_name, country_name_to_process])
        
        self.df[['valid_already', 'to_process']] = self.df.apply(split_valid_toprocess, axis=1)
        
        self.valid_already_df = self.df[self.df['to_process'].str.len() == 0].copy()
        self.to_process_df = self.df[self.df['to_process'].str.len() > 0].copy()
    
        logger.info(f'Country name standardization: {len(self.to_process_df)} out of {len(self.df)} data need processing')

    
    def llm_process(self, llm) -> None:
        """ 
        Run llm on 'to_process' column of to_process_df, the result is saved in 'country_clean' column. 

        This method update: 
        - self.to_process_df: new column 'country_clean' containing llm-processed country names appended. 
        """

        temp_path = self.output_dir / "llm_stdname_temp.csv"

        self.to_process_df, temp_path = task_run(self.to_process_df, 
                                                 llm, 
                                                 self.prompt, 
                                                 'to_process',                 
                                                 self.output_column,
                                                 temp_path)  


        self.to_remove.append(temp_path)


    def postprocess(self) -> None:
        """ 
        Check the terms in country_clean column, whether they are in std_country_set, although explicitly required in the prompt.
        Then add the term in valid_already colum to country_clean, then deduplicate. 

        This methods updates:
        - self.to_process_df: llm-processed 'country_clean' is double checked.
        - self.df: concatenate to_process_df and valid_already_df back together. 
        """

        self.to_process_df[self.output_column] = self.to_process_df[self.output_column].apply(lambda x: self.preprocess_single(str(x), self.replacement_maps, self.frequent_unsupported))
        self.to_process_df[self.output_column] = self.to_process_df[self.output_column].apply(lambda x: [c for c in x if c in self.std_country_set])

        self.to_process_df[self.output_column] = self.to_process_df['valid_already'] + self.to_process_df[self.output_column]
        self.to_process_df[self.output_column] = self.to_process_df[self.output_column].apply(lambda x: list(set(x)))

        self.valid_already_df.loc[:,self.output_column] = self.valid_already_df['valid_already']

        self.df = pd.concat([self.to_process_df, self.valid_already_df])

        self.df = self.df[self.result_column_list]
        self.df = self.df.sort_index()


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

        if pd.isna(country_str) or country_str == 'None':
            return []

        countries = read_list_string_from_df(country_str)

        if frequent_unsupported:
            countries = [c for c in countries if c not in frequent_unsupported]

        if replacement_maps:
            for re_map in replacement_maps.values():
                countries = [re_map.get(c, c) for c in countries]
    
        return countries




