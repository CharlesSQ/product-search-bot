import json
import re
from langchain.schema import BaseOutputParser
# from langchain.output_parsers.json import _custom_parser
from typing import Any


class ConversationAgentOutputParser(BaseOutputParser):
    """
    Parses the output of an LLM call that is expected to be a JSON object.

    This parser specifically looks for a JSON structure containing 'action' and
    'action_input' keys, which are standard for LangChain agents.
    """

    def parse(self, text: str) -> Any:
        """
        Parses the input text, expecting a JSON object with agent action details.

        Args:
            text (str): The raw string output from the language model.

        Returns:
            Any: A dictionary containing 'action' and 'action_input' if parsing is
                 successful. Otherwise, a dictionary with an 'error' key.
        """
        try:
            data = json.loads(text)
            if "action" in data and "action_input" in data:
                return data
        except json.JSONDecodeError:
            pass
        return {"error": "not valid json"}
