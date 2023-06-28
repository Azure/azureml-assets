"""File for the ConstructPrompt Class."""
from constants import TaskTypes


class ConstructPrompt:
    """Class for creating Prompts."""

    def __init__(self, qa_task):
        """Initialize the class."""
        if qa_task == TaskTypes.arithmetic:
            self.construct_prompt = self.construct_arithmetic_prompt
        if qa_task == TaskTypes.mcq:
            self.construct_prompt = self.construct_mcq_prompt
        if qa_task == TaskTypes.abstractive:
            self.construct_prompt = self.construct_extractive_prompt

    def construct_arithmetic_prompt(self, question, meta_prompt, context=None):
        """Build a Arithmetic Prompt."""
        prompt = "Question: "+question
        prompt += "\nA: "+meta_prompt
        return prompt

    def construct_mcq_prompt(self, question, meta_prompt=None, context=None, phase=1):
        """Build a Multi Choice Question Prompt."""
        prompt = "Question: "+question
        prompt += "\nA: "+meta_prompt
        return prompt

    def construct_extractive_prompt(self, question, meta_prompt=None, context=None, phase=1):
        """Build a Extractive Prompt."""
        prompt = meta_prompt
        prompt += "\n"+question
        prompt += "\nAnswer:"
        return prompt
