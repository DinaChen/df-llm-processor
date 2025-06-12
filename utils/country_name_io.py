import yaml
import pandas as pd
import ast
import re
from typing import Union, Set
from pathlib import Path


def read_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def read_list_string_from_df(val):
    """
    Parse firm_export_country to a flat list of strings.
    Handles: NaN, list, stringified list, comma/Chinese-comma/顿号 separated, and single string.
    """
    if pd.isna(val):
        return []
    # If already a list, flatten recursively
    if isinstance(val, list):
        result = []
        for item in val:
            result.extend(read_list_string_from_df(item))
        return result
    # Try to parse string as list
    if isinstance(val, str):
        val = val.strip()
        # Try to parse stringified list
        if val.startswith("[") and val.endswith("]"):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    result = []
                    for item in parsed:
                        result.extend(read_list_string_from_df(item))
                    return result
            except Exception:
                pass
        # If not a list, split by common delimiters
        return [v.strip() for v in re.split(r'[,\uFF0C\u3001;；]', val) if v.strip()]
    # Fallback: convert to string
    return [str(val).strip()] if str(val).strip() else []
"""
def read_dicts_from_yaml(yaml_file):

    if isinstance(yaml_file, dict):
        return yaml_file
    
    if yaml_file:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    return None
"""


def read_dicts_from_yaml(dictionary: Union[Path, dict, None]) -> Union[dict, None]:
    """
    Safely dictionary from a Path (YAML).
    Returns a dictionary or None.
    """
    if isinstance(dictionary, Path):
        try:
            with open(dictionary, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading dictonary from {dictionary}: {e}")
            return None
    elif isinstance(dictionary, dict):
        return dictionary
    else:
        return None
    


def read_standard_country(std_country_set: Union[Path, set]) -> Set[str]:
    """
    Read standard country name set from a Path or return the set directly.
    Returns a set of country names.
    """
    if isinstance(std_country_set, Path):
        try:
            std_list = pd.read_excel(std_country_set)
            return set(std_list['country'].dropna())
        except Exception as e:
            print(f"Error reading standard country name set from {std_country_set}: {e}")
            return set()
    elif isinstance(std_country_set, set):
        return std_country_set
    else:
        raise ValueError("std_country_set must be a Path or a set")