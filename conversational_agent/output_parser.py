import json
import re
from langchain.schema import BaseOutputParser
# from langchain.output_parsers.json import _custom_parser
from typing import Any


class ConversationAgentOutputParser(BaseOutputParser):

    def parse(self, text: str) -> Any:
        """
        Gets the Json object from the output of the LLM.
        """
        # try:
        #     response = parse_json_markdown(text)

        #     return response

        # except Exception:

        #     return {"error": "not valid json"}
        try:
            data = json.loads(text)
            if "action" in data and "action_input" in data:
                return data
        except json.JSONDecodeError:
            pass
        return {"error": "not valid json"}


# def parse_json_markdown(json_string: str) -> dict:
#     """
#     Parse a JSON string from a Markdown string.

#     Args:
#         json_string: The Markdown string.

#     Returns:
#         The parsed JSON object as a Python dictionary.
#     """
#     # Try to find 2 JSON string within triple backticks
#     two_matches = re.search(r"```(json)?(.*)```(.*)```(json)?(.*)```",
#                             json_string, re.DOTALL)

#     # Try to find 1 JSON string within triple backticks
#     one_match = re.search(r"```(json)?(.*)```", json_string, re.DOTALL)

#     # If no match found, assume the entire string is a JSON string
#     json_str = ''

#     if two_matches:
#         json_str = two_matches.group(5)
#     elif one_match:
#         # If match found, use the content within the backticks
#         json_str = one_match.group(2)
#     else:
#         # Raise error
#         raise ValueError("No JSON string found in input.")

#     # Strip whitespace and newlines from the start and end
#     json_str = json_str.strip()

#     # handle newlines and other special characters inside the returned value
#     json_str = _custom_parser(json_str)

#     # Parse the JSON string into a Python dictionary
#     parsed = json.loads(json_str)

#     return parsed
