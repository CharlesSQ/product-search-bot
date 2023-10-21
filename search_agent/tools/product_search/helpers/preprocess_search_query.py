# preprocess_search_query.py
from .utils import vectorize_JSON_list, create_embedding
# Add any other necessary imports here, such as JSON handling, etc.


def preprocess_search_query(search_query, extraction_chain):
    max_retries = 5  # Max number of times to retry
    retries = 0  # Counter for number of retries

    while retries < max_retries:
        if len(search_query.split(" ")) <= 1:
            search_query = f'good for {search_query}'

        # Use the extraction_chain passed as an argument
        search_query_JSON_list = extraction_chain.run(search_query)

        # No need to convert to dictionary here; assume it's a list of non-empty JSON objects
        if search_query_JSON_list:
            break  # Break the loop if a list of JSON objects is obtained

        retries += 1  # Increment the retry counter

        if retries == max_retries:
            raise ValueError(
                "Reached maximum number of retries without obtaining a non-empty list of JSON objects.")

    # Vectorization logic
    vectorized_search_query_JSON_list = vectorize_JSON_list(
        search_query_JSON_list)

    search_query_JSON_string = str(search_query_JSON_list) + search_query
    weighted_query_vector = create_embedding("", search_query_JSON_string)

    return search_query_JSON_list, search_query_JSON_string, vectorized_search_query_JSON_list, weighted_query_vector
