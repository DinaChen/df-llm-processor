# The 
# A LLmTaskProcessor works on a dataframe, take one or multiple columns
# Perform pre-processing then run llm to get response
# Return the same dataframe with appended new result column. 
from abc import ABC, abstractmethod

class LlmTaskProcessor(ABC):

    @abstractmethod
    def preprocess(self, df, *args, **kwargs):
        """
        Preprocess the input dataframe.
        """
        pass

    @abstractmethod
    def llm_process(self, df, *args, **kwargs):
        """
        Run task with llm
        """
        pass

    @abstractmethod
    def postprocess(self, df, *args, **kwargs):
        """
        Postprocess the dataframe after LLM inference.
        """
        pass
