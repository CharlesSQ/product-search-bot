from scipy.stats import hmean
from collections import Counter
from pprint import PrettyPrinter
from langchain.chat_models import ChatOpenAI
import openai
from config import MODEL

# Initialize your language model
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
import sys
import pprint

sys.setrecursionlimit(3000)

pp = pprint.PrettyPrinter(depth=6)


# // Custom functions and logic that Langchain doesn't have //
# Function to vectorize a list of JSON objects (which are dictionaries in Python)
def vectorize_JSON_list(search_query_JSON_list):
    vectorized_list = []
    for single_JSON in search_query_JSON_list:
        vectorized_list.append(vectorize_dict(single_JSON))
    return vectorized_list

# Function to vectorize a dictionary
def vectorize_dict(search_query_dict):
    vectorized_dict = {}
    for key, value in search_query_dict.items():
        if isinstance(value, dict):
            vectorized_dict[key] = vectorize_sub_dict(value)
        else:
            vectorized_dict[key] = create_embedding(key, value)
    return vectorized_dict

# Function to vectorize sub-dictionary
def vectorize_sub_dict(sub_dict):
    return {k: create_embedding(k, v) for k, v in sub_dict.items()}

# Function to create embedding
def create_embedding(key, value):
    input_string = f"{key} {' '.join(map(str, value))}" if isinstance(value, list) else f"{key} {str(value)}"
    return openai.Embedding.create(input=input_string, engine=MODEL)['data'][0]['embedding']