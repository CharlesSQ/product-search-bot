from dotenv import load_dotenv
from pymongo import MongoClient
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain

from .helpers.config import *
from .helpers.pinecone_helpers import *
from .helpers.mongo_helpers_v2 import *
from .helpers.mongo_helpers.extract_labels import *
from .helpers.skincare_labels import skincare_schema
from .helpers.pcsearch import search, parallel_search
from .helpers.preprocess_search_query import preprocess_search_query

import os
load_dotenv()
openai_api_key = os.environ['OPENAI_API_KEY']

# This works locally well and fast. Upload those two JSON files into local MongoDB as new collections under the database "Mecca"


class ProductSearchTool():
    def __init__(self):
        # Establish a client connection to your MongoDB instance
        client = MongoClient('mongodb://localhost:27017/')

        # Access your database and collection.
        self.db = client['Mecca']

    def products_retriever(self):
        # Collection in MongoDB which has the label counts of products
        product_label_summaries = self.db['deletethreebruh']
        # Collection of review labels that have a positive sentiment, their vectorized format, and their corresponding master labels
        # positive_labels = self.db['positive_labels']

        # Access environment variables

        print(f"OpenAI API Key: {openai_api_key}")

        # Initialize the language model
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

        # Your existing code
        # search_query = """fix rosacea dry skin under $100"""
        search_query = """combat pigmentation for combo skin. Im 29"""

        # Initialize your skincare-specific chain
        # Need to inject my own prompt into here somehow for refined outputs
        skincare_search_query_chain = create_extraction_chain(
            skincare_schema, llm, verbose=True)

        # Use function which vectorizes each key within the skincare-specific chain.

        search_query_JSON_list, search_query_JSON_string, vectorized_search_query_JSON_list, weighted_query_vector = preprocess_search_query(
            search_query, skincare_search_query_chain)

        # # Need to add logic so that if the category does not exist in the existing refined_category or master_category or brand etc, then it just goes ahead and executes the search query normally.
        # # These should be retrieved in the refined_category collections of MongoDB
        filter_criteria = {}

        # Set top_k_values and call parallel_search function
        # These are the amount of search results for each query
        top_k_values = [10, 100, 500]
        search_results = parallel_search(index, weighted_query_vector, top_k_values,
                                         filter_criteria, metadata_key='type', desired_value='prod', namespace='products')
        # Choose the search result with the desired top_k value
        filtered_res = search_results[2]

        # Grabbing list of unique product ids from the product results
        prod_ids = extract_product_ids(filtered_res)
        unique_prod_ids = list(set(prod_ids))

        # Create a dictionary of products by calling the MongoDB collection using the unique product ids
        docs_dict = retrieve_documents(
            product_label_summaries, unique_prod_ids)

        # Constants for label searching
        metadata_key = 'type'
        desired_value_label = 'label'
        top_k = 100
        filter_criteria_positive = {'sentiment': 'positive'}

        # 1. Search for top labels relevant to search query
        search_query_labels = search(index, weighted_query_vector, metadata_key,
                                     desired_value_label, top_k, filter_criteria_positive, namespace='')
        # Extract the list of ids
        search_query_labels_id_list = extract_ids(search_query_labels)
        print(f"search_query_labels_id_list: {search_query_labels_id_list}")

        # 2. Search for top labels relevant to vectorized search queries
        top_k_labels_search_params = 40

        results_list = []

        for query_dict in vectorized_search_query_JSON_list:
            for key, query_vector in query_dict.items():
                result = search(index, query_vector, metadata_key, desired_value_label,
                                top_k_labels_search_params, filter_criteria_positive, namespace='')
                results_list.append({key: result})

        # Create a list of dictionaries, each containing the original query and extracted ids
        search_query_params_ids_dict = extract_ids_from_results_list(
            results_list)
        print(f"search_query_params_ids_dict: {search_query_params_ids_dict}")

        # Builds a database of master label dictionaries and unique master labels for later use
        print("Building base list of master labels and unique labels...\n")
        master_label_dict, unique_master_labels = define_parameters_and_rerank_labels(
            self.db)

        # Outputting processed information for debugging or logs
        print(f"Master Label Dictionary in Database: {master_label_dict}")
        print("=" * 70)
        print(f"Unique Master Labels in Database: {unique_master_labels}")
        print("=" * 70)

        # Processing label IDs related to the search query using the function from another file
        print("Processing search query labels...\n")
        search_query_filtered_labels, search_query_master_label_list = process_id_label_list(
            self.db, search_query_labels_id_list, master_label_dict, unique_master_labels)

        # Outputting processed information for debugging or logs
        print(
            f"Search Query Master Label List: {search_query_master_label_list}")

        ######################################################################################################

        # Process search query parameters to perform label matching using function from another file
        print("Processing search_query_params_flattened_unique_labels, list of label ids related to the search query parameters...\n")
        search_query_param_labels_in_master_label_dict = process_search_query_params_ids_dict(
            self.db, search_query_params_ids_dict)

        # Output processed information for debugging or logs
        print(
            f"Flattened Unique Master Labels for Search Query Parameters: {search_query_param_labels_in_master_label_dict}")

        ######################################################################################################

        # Process fuzzy matching to perform label matching against all parameters from the search_query_dict

        print("Performing fuzzy matching on search query parameters...\n")
        _, _, search_query_params_fuzzy_master_labels = perform_fuzzy_matching_for_attributes(
            search_query_JSON_list, search_query_filtered_labels, master_label_dict, threshold=90)

        # Output processed information for debugging or logs
        print(
            f"Master Labels Based on Fuzzy-Matched Labels: {search_query_params_fuzzy_master_labels}")

        ######################################################################################################

        # Get final list of labels and their weights, by processing them with openAI
        final_list = get_weighted_unique_list(
            search_query_master_label_list,
            search_query_param_labels_in_master_label_dict,
            search_query_params_fuzzy_master_labels
        )
        print(f"final_list: {final_list}")

        master_labels_extracted_using_AI_and_schema = extract_master_label_weights_using_openAI(
            search_query, final_list[:10], 4)
        print(
            f"master_labels_extracted_using_AI_and_schema: {master_labels_extracted_using_AI_and_schema}")

        # Initialize an empty list to store processed product data
        product_data_with_relevance = []

        # Assuming filtered_res['matches'] contains the list of dictionaries
        product_search_results = filtered_res['matches']

        for product_metadata in product_search_results:  # Loop over product_search_results
            try:
                product_data_with_relevance_entry = process_product(
                    product_metadata,
                    docs_dict,
                    master_labels_extracted_using_AI_and_schema,
                    search_query_JSON_list,
                    master_label_dict
                )
                # Removing products that don't have reviews (i.e. the ids don't appear in the mongoDB review_summaries collection)
                if product_data_with_relevance_entry is not None:
                    product_data_with_relevance.append(
                        product_data_with_relevance_entry)
            except Exception as e:
                print(
                    f"Error processing product {product_metadata.get('id', 'unknown_id')}: {e}")
                import traceback
                # This will print the stack trace, showing you exactly where the error occurred.
                traceback.print_exc()

        # Assuming product_data_with_relevance is a list of dictionaries

        # Sort by having "Relevant to You" data and by descending Relevance Score
        sort_products(product_data_with_relevance)

    def get_tool(self):
        return Tool(
            name="search_products",
            description="Searches and returns information regarding products in the database.",
            func=self.products_retriever

        )


def custom_sort(x):
    """
    Custom sorting function for the list of products
    """
    if x is None:
        return (0, 0)  # Lowest priority for None
    relevant_to_you_str = x.get(
        'filtered_master_label_counts_using_AI_dict_str', 'N/A')
    has_relevant_data = 1 if relevant_to_you_str and relevant_to_you_str != 'N/A' else 0
    return (has_relevant_data, x.get('relevance_score', 0))


def sort_products(product_data_with_relevance, custom_sort):
    """
    Sorts the list of products by relevance score and whether the product has relevant data
    """
    sorted_products = sorted(product_data_with_relevance,
                             key=custom_sort, reverse=True)

    for index, product in enumerate(sorted_products):
        if product is not None:
            metadata = product.get('product_metadata', {}).get(
                'metadata', {})  # Adjusted line here

            print(f"--- Product {index + 1} ---")
            print(
                f"Relevance Score: {product.get('relevance_score', 'N/A')}")
            # Adjusted line here
            print(f"Title: {metadata.get('title', 'N/A')}")
            # Adjusted line here
            print(f"Brand: {metadata.get('brand', 'N/A')}")
            # Adjusted line here
            print(f"Price: ${metadata.get('price', 'N/A')}")
            # Adjusted line here
            print(
                f"Master Category: {metadata.get('master_category', 'N/A')}")
            # Adjusted line here
            print(
                f"Refined Category: {metadata.get('refined_category', 'N/A')}")
            # Adjusted line here
            print(f"Product Link: {metadata.get('prod_link', 'N/A')}")
            print(f"Reviews Count: {product.get('reviews_count', 'N/A')}")

            relevant_to_you_str = product.get(
                'filtered_master_label_counts_using_AI_dict_str', 'N/A')
            if relevant_to_you_str == 'N/A' or not relevant_to_you_str.strip():
                print("Relevant to You: No labels matched")
            else:
                print(f"Relevant to You: {relevant_to_you_str}")

            master_label_counts_str = product.get(
                'master_label_counts_str', 'N/A')
            if master_label_counts_str != 'N/A':
                # Notice the change in the split delimiter to ' • '
                first_five_values = ' • '.join(
                    master_label_counts_str.split(' • ')[:5])
                print(f"Review Overview: {first_five_values}")
            else:
                print("Review Overview: N/A")

            print("\n" + "="*40)
        else:
            print(f"\n--- Product {index + 1} is None ---")
