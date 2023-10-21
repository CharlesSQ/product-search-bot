from bson.objectid import ObjectId
from collections import Counter
import pandas as pd
from fuzzywuzzy import process
from collections import defaultdict
from fuzzywuzzy import fuzz
from scipy.stats import hmean
import numpy as np
from .mongo_helpers.extract_labels import extract_master_labels_using_openAI
from dateutil.parser import parse
from datetime import datetime, date
# Make sure to import the appropriate classes from your boosters.py file
from .mongo_helpers.boosters import TitleBooster, CategoryBooster
import pymongo
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Function to find top master label matches (returns top 3 by default if limit not called)


def get_top_master_label_matches(skin_concerns, master_label_list, limit=3):
    # Check if skin_concerns is None and return an empty list if it is
    if skin_concerns is None:
        return []

    master_check = []
    for concern in skin_concerns:
        matches = process.extract(concern, master_label_list, limit=limit)
        master_check.extend([match[0] for match in matches])
    return master_check

# Defining function for fuzzy scores


def get_fuzzy_scores(input_values, labels, min_score=50):
    if input_values is None:
        input_values = []

    fuzzy_scores = defaultdict(int)

    for concern in input_values:
        for label in labels:
            match_score = fuzz.token_set_ratio(label, concern)
            if match_score > min_score and match_score > fuzzy_scores[label]:
                fuzzy_scores[label] = match_score

    return fuzzy_scores


def calculate_base_relevance_score(sorted_positive_labels_in_doc, reviews_count):
    total_label_count = sum(
        label_dict['reviews'] for label_dict in sorted_positive_labels_in_doc.values())
    base_relevance_score = total_label_count / \
        reviews_count if reviews_count > 0 else 0
    return base_relevance_score


def get_master_label_counts(positive_labels_in_doc, master_label_dict):
    master_label_counts = {}
    for original_label, label_dict in positive_labels_in_doc.items():
        master_label = master_label_dict.get(original_label, original_label)
        if master_label not in master_label_counts:
            master_label_counts[master_label] = 0
        master_label_counts[master_label] += label_dict['reviews']

    # Create the string representation
    master_label_counts_str = ' • '.join(
        f'{key} ({value})' for key, value in master_label_counts.items())

    return master_label_counts, master_label_counts_str


def convert_label_totals_to_str(master_label_totals):
    master_label_total_labels = [{'label': master_label, 'reviews': count}
                                 for master_label, count in master_label_totals.items()]
    return " • ".join([f"{label_dict['label']} ({label_dict['reviews']})" for label_dict in master_label_total_labels])


def filter_label_counts(master_label_counts, matching_labels):
    return [{'label': key, 'reviews': value} for key, value in master_label_counts.items() if key in matching_labels]


def calculate_total_boost(filtered_label_counts, reviews_count, search_query_master_labels_weights=None):
    total_boost = 0

    # Check if search_query_master_labels_weights is a list of 3-tuples
    if all(isinstance(item, tuple) and len(item) == 3 for item in search_query_master_labels_weights or []):
        weight_dict = {label: weight for label, _,
                       weight in search_query_master_labels_weights}
    elif search_query_master_labels_weights is None:
        weight_dict = {label_dict['label']: 1 for label_dict in filtered_label_counts}
    else:
        raise ValueError(
            "search_query_master_labels_weights must be a list of 3-tuples.")

    # Ensure each item in filtered_label_counts is a dictionary with a 'label' key
    if not all(isinstance(label_dict, dict) and 'label' in label_dict for label_dict in filtered_label_counts):
        raise ValueError(
            "Each item in filtered_label_counts must be a dictionary with a 'label' key.")

    for label_dict in filtered_label_counts:
        label = label_dict['label']
        # Default to 1 if label has no weight
        label_weight = weight_dict.get(label, 1)
        label_percentage = label_dict['reviews'] / reviews_count
        harmonic_mean = hmean(
            [label_dict['reviews'], 1 / (label_percentage + 1e-9)])

        # Print calculation details
        print(f"Label: {label}")
        print(f"Label Weight: {label_weight}")
        print(f"Label Percentage: {label_percentage}")
        print(f"Harmonic Mean: {harmonic_mean}")
        print(f"Weighted Harmonic Mean: {harmonic_mean * label_weight}")
        print("----------------------------")

        total_boost += harmonic_mean * label_weight  # Apply the weight here

    print(f"Total Boost: {total_boost}")
    return total_boost


def filter_label_counts(overview_labels, label_list_most_relevant_to_search_concerns):
    return [{'label': label, 'reviews': overview_labels[label]} for label in label_list_most_relevant_to_search_concerns if label in overview_labels]


def get_mapped_labels(label_list_most_relevant_to_search_concerns, master_label_dict):
    print(f"Master label dict: {master_label_dict}")  # Debugging line
    # Debugging line
    print(
        f"Entered get_mapped_labels with {label_list_most_relevant_to_search_concerns}")

    # Initialize an empty list for mapped labels
    mapped_labels = []

    # Loop through each label to map it using master_label_dict
    for label in label_list_most_relevant_to_search_concerns:
        # Find the key whose value is 'label'
        key_for_label = [key for key,
                         value in master_label_dict.items() if value == label]

        if key_for_label:
            mapped_labels.append(label)
        else:
            # Debugging line
            print(f"Label {label} not found in master_label_dict")

    print(f"Mapped labels: {mapped_labels}")  # Debugging line

    # Count the occurrence of each label
    label_counts = Counter(mapped_labels)
    print(f"Label counts: {label_counts}")  # Debugging line

    # Compute the total number of labels
    total_labels = sum(label_counts.values())
    print(f"Total labels: {total_labels}")  # Debugging line

    # Create a list of tuples where each tuple is (label, occurrence, weighting)
    # and sort the list by occurrence in descending order
    search_query_master_labels_weights = sorted(
        [(label, count, count / total_labels) for label, count in label_counts.items()], key=lambda x: x[1], reverse=True)
    # Debugging line
    print(f"Sorted label weights: {search_query_master_labels_weights}")

    # Create a list of just the labels
    search_query_master_labels = [
        label for label, count, weight in search_query_master_labels_weights]

    return search_query_master_labels_weights, search_query_master_labels


def get_original_labels_for_master_labels(master_labels, master_label_dict):
    original_labels_dict = {}
    for original, master in master_label_dict.items():
        for master_label in master_labels:
            # Check if master_label is a tuple, and if so, take the first element
            if isinstance(master_label, tuple):
                master_label = master_label[0]

            if master == master_label:
                if master not in original_labels_dict:
                    original_labels_dict[master] = []
                original_labels_dict[master].append(original)
    return original_labels_dict


def get_matching_labels(original_labels_dict, overview_labels):
    product_labels_matched_to_search_query = []
    for original_labels in original_labels_dict.values():
        for label in original_labels:
            if label in overview_labels:
                product_labels_matched_to_search_query.append(
                    (label, overview_labels[label]))
    return product_labels_matched_to_search_query


def extract_exact_master_labels(search_query_master_label_list, labels):
    extracted_labels = [
        label for label in search_query_master_label_list if label in labels]
    return extracted_labels


def unique_ordered(values):
    seen = set()
    unique_values = []
    for value in values:
        if value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values


def add_skin_type_labels_to_search_query_labels(extracted_labels_from_clustered_labels, master_label_counts, product_labels_matched_to_search_query):
    for label in extracted_labels_from_clustered_labels:
        if label in master_label_counts:
            # Accessing 'reviews' count from the inner dictionary of master_label_counts
            count = master_label_counts[label]['reviews']
            # Appending tuple (label, count) to product_labels_matched_to_search_query
            product_labels_matched_to_search_query.append((label, count))
    return product_labels_matched_to_search_query


def extract_matched_query_to_skin_type_overviews(clustered_labels, labels_in_doc):
    matched_labels = []
    for label in clustered_labels:
        for doc_label, associated_labels in labels_in_doc:
            if label == doc_label:
                matched_labels.append((doc_label, associated_labels))
    return matched_labels


def get_labels_from_ids(db, collection_name, ids_input):
    """
    Given a database, collection name and a list or dictionary of ids, 
    fetches the corresponding labels from the database.

    Args:
    db (pymongo.database.Database): The database to fetch labels from.
    collection_name (str): The collection from which to fetch the labels.
    ids_input (list of str or dict of str: list of str): A list or dictionary where each value 
                                                         is a list of ids.

    Returns:
    list of str or dict of str: list of str: A list or dictionary where each key corresponds to a key from ids_input 
                                                   and each value is a list of 'label' values from the database for the corresponding ids.
    """

    def fetch_labels(id_list):
        labels = []
        for id in id_list:
            # print(f"Attempting to convert ID of type {type(id)}: {id}")
            try:
                if not isinstance(id, (str, bytes, ObjectId)):
                    print(f"Skipping invalid ID of type {type(id)}: {id}")
                    continue
                doc = db[collection_name].find_one(
                    {"_id": ObjectId(id)}, {"label": 1, "_id": 0})
                if doc is not None:
                    labels.append(doc['label'])
            except Exception as e:
                print(f"An error occurred: {e}")

        return labels

    # Initialize an empty dictionary to store the output labels
    labels_output = {}

    # Check if ids_input is a list of dictionaries
    if isinstance(ids_input, list) and all(isinstance(d, dict) for d in ids_input):
        for single_dict in ids_input:
            for key, id_list in single_dict.items():
                labels_output[key] = fetch_labels(id_list)
    elif isinstance(ids_input, dict):
        labels_output = {key: fetch_labels(id_list)
                         for key, id_list in ids_input.items()}
    elif isinstance(ids_input, list):
        labels_output = fetch_labels(ids_input)
    else:
        raise TypeError(
            "Invalid type for ids_input. Must be a dictionary or a list.")

    return labels_output


def create_label_master_dict(db, collection_name, query):
    """
    Create a dictionary for labels and their master labels based on the given query.

    Args:
    db (pymongo.database.Database): The database to fetch labels from.
    collection_name (str): The collection from which to fetch the labels.
    query (dict): The query to be executed on the MongoDB collection.

    Returns:
    dict: A dictionary mapping labels to their master labels.
    """
    label_master_dict = {}
    try:
        # Add a debug print statement
        print("Executing query...")

        # Add a 5-second timeout for query execution with max_time_ms=5000
        cursor = db[collection_name].find(
            query, cursor_type=pymongo.CursorType.EXHAUST, max_time_ms=5000)

        # Another debug print statement
        print("Query executed, processing results...")

        # Process the result from MongoDB query
        label_master_dict = {doc['label']: doc['master_label']
                             for doc in cursor if doc['label'] is not None and doc['master_label'] is not None}

        # Debug print statement indicating end of processing
        print("Results processed.")

        # Clean-up NaNs (considering you might want to use pandas for this)
        label_master_dict = {k: v for k, v in label_master_dict.items() if pd.notna(
            k) and pd.notna(v) and str(k) != 'nan' and str(v) != 'nan'}

    except PyMongoError as e:
        print(
            f"An error occurred while fetching or processing data from MongoDB: {e}")

    return label_master_dict


def get_unique_labels(master_label_dict, labels):
    """
    Create a list or dictionary of unique labels based on the master_label_dict and labels.

    Args:
    master_label_dict (dict): A dictionary mapping labels to their master labels.
    labels (list or dict): The list or dictionary of label categories, which may include tuples.

    Returns:
    list or dict: A list or dictionary of unique labels.
    """

    # Function to process label list or list of tuples
    def process_labels(label_list):
        # If the input is a list of tuples, extract only the label strings
        if all(isinstance(item, tuple) for item in label_list):
            label_list = [label for label, _ in label_list]

        # Use the master label if it exists; otherwise, use the original label
        clustered_labels = [master_label_dict.get(
            label, label) for label in label_list]

        # Get unique labels by converting the list to a dictionary and back to a list
        return list(dict.fromkeys(clustered_labels))

    # If labels is a dictionary, process each category separately
    if isinstance(labels, dict):
        unique_labels_by_category = {category: process_labels(
            category_labels) for category, category_labels in labels.items()}
        return unique_labels_by_category

    # If labels is a list, process the whole list
    else:
        return process_labels(labels)


def combine_and_average(lists, weights):
    combined_dict = {}
    for i, sublist in enumerate(lists):
        for label, num, weight in sublist:
            list_weight = weights[i]
            if label not in combined_dict:
                combined_dict[label] = [num, weight * list_weight, 1]
            else:
                combined_dict[label][0] += num
                combined_dict[label][1] += weight * list_weight
                combined_dict[label][2] += 1

    averaged = []
    for label, values in combined_dict.items():
        avg_weight = values[1] / values[2]
        averaged.append((label, values[0], avg_weight))

    # Sort the 'averaged' list in descending order of average weight
    averaged.sort(key=lambda x: x[2], reverse=True)

    # Generate 'labels_only' from 'averaged' to ensure the same order
    labels_only = [item[0] for item in averaged]

    return averaged, labels_only


def apply_weights_to_labels(label_input, weight_dict_labels=None, start_weight=1.0, decay_rate=0.8, min_weight=0.01, equal_weighting=False):
    # print(f"Debug: label_input = {label_input}")
    # print(f"Debug: weight_dict_labels = {weight_dict_labels}")
    # print(f"Debug: equal_weighting = {equal_weighting}")
    weighted_labels = []

    if equal_weighting:
        # Extract labels whether input is dict or list
        if isinstance(label_input, dict):
            for key in label_input:
                for label in label_input[key]:
                    weighted_labels.append(label)
        elif isinstance(label_input, list):
            weighted_labels.extend(label_input)

        # Determine the equal weight for each label
        num_labels = len(weighted_labels)
        equal_weight = 1.0 / num_labels if num_labels != 0 else 0

        # Assign the equal weight to each label
        weighted_labels = [(label, 1, equal_weight)
                           for label in weighted_labels]
        return weighted_labels

    # Original logic
    if isinstance(label_input, dict):  # For dictionary input
        for key in label_input:
            # Get the start_weight and decay_rate for this key, or use a default if it's not defined
            key_start_weight, key_decay_rate = weight_dict_labels.get(
                key, (0.5, 0.05))
            weight = key_start_weight
            for label in label_input[key]:
                # Prevent weight from dropping below min_weight
                weight = max(weight, min_weight)
                weighted_labels.append((label, 1, weight))
                # Decrease the weight for the next iteration
                weight = max(weight - key_decay_rate, min_weight)
    elif isinstance(label_input, list):  # For list input
        weight = start_weight
        for label in label_input:
            # Prevent weight from dropping below min_weight
            weight = max(weight, min_weight)
            weighted_labels.append((label, 1, weight))
            # Decrease the weight for the next iteration
            weight = max(weight - decay_rate, min_weight)

    # Normalize the weights so they sum to 1
    total_weight = sum(x[2] for x in weighted_labels)
    weighted_labels = [(x[0], x[1], x[2]/total_weight)
                       for x in weighted_labels]

    # You could sort the labels here by weight if you want
    weighted_labels.sort(key=lambda x: x[2], reverse=True)
    return weighted_labels


def filter_master_label_counts_vs_AI_search_weights(master_label_counts, weight_tuples):
    filtered_master_label_counts_using_AI_dict = []
    for weight_tuple in weight_tuples:
        label = weight_tuple[0]  # Accessing the first element of the tuple
        # Check if the label is found in master_label_counts
        if label in master_label_counts:
            # Directly get the count as it's an integer
            count = master_label_counts[label]
            filtered_master_label_counts_using_AI_dict.append(
                {'label': label, 'reviews': count})

    filtered_master_label_counts_using_AI_dict_str = ' • '.join(
        [f"{item['label']} ({item['reviews']})" for item in filtered_master_label_counts_using_AI_dict])
    return filtered_master_label_counts_using_AI_dict, filtered_master_label_counts_using_AI_dict_str


def retrieve_documents(collection, unique_prod_ids):
    return {doc['prod_id']: doc for doc in collection.find({'prod_id': {'$in': unique_prod_ids}})}

############################ Weight application to labels & using AI to extract most relevant labels after MongoDB retrieval ##########################################


def define_parameters_and_rerank_labels(db):
    """
    Define search parameters and rerank labels based on a database query.

    This function performs the following operations:
    1. Calls the create_label_master_dict function to build a master label dictionary from the database.
    2. Creates a unique list of master labels to be used later for ranking and filtering.

    Args:
        db: The database connection object.

    Returns:
        Tuple containing:
        - Dictionary of master labels mapped from the database ('master_label_dict').
        - List of unique master labels ('unique_master_labels').
    """

    # Create a master label dictionary from the database
    master_label_dict = create_label_master_dict(
        db, 'positive_labels', {"master_label": {"$ne": np.nan}})

    # Create a unique list of master labels
    unique_master_labels = list(set(master_label_dict.values()))

    return master_label_dict, unique_master_labels


def process_id_label_list(db, id_list, master_label_dict, unique_master_labels):
    """
    Process a list of label IDs to perform several tasks.

    This function:
    1. Retrieves labels corresponding to a list of given IDs.
    2. Obtains unique labels from these retrieved labels.
    3. Creates a master dictionary for MongoDB query.
    4. Filters out unique labels that already exist in a master list of unique labels.

    Args:
        db: The database connection.
        id_list: List of label IDs to be processed.
        master_label_dict: Dictionary of master labels.
        unique_master_labels: List of unique master labels.

    Returns:
        Tuple containing:
        - List of labels filtered based on the given IDs.
        - List of unique labels among the filtered labels.
        - List of unique labels that also exist in the master list of unique labels.
    """

    # Get labels corresponding to the given IDs
    filtered_labels = get_labels_from_ids(db, 'positive_labels', id_list)

    # Obtain unique labels
    unique_labels = get_unique_labels(master_label_dict, filtered_labels)

    # Create ObjectIds from id_list for MongoDB querying
    object_ids = [ObjectId(id) for id in id_list]

    # Create label master dict for MongoDB
    master_label_dict_label_search_query = create_label_master_dict(
        db, 'positive_labels', {"_id": {"$in": object_ids}})

    # Obtain unique labels from search query
    unique_labels_search_query = get_unique_labels(
        master_label_dict_label_search_query, filtered_labels)

    # Filter out unique labels that exist in unique_master_labels
    search_query_labels_in_master_label_dict = [
        label for label in unique_labels_search_query if label in unique_master_labels]

    return unique_labels_search_query, search_query_labels_in_master_label_dict


def process_search_query_params_ids_dict(db, search_query_params_ids_dict):
    """
    Process and filter labels based on search query parameters.

    Given a dictionary of search query parameters, this function performs the following:
    1. Retrieves filtered labels based on the given search query parameters.
    2. Flattens and finds unique labels from the filtered labels.
    3. Creates a master dictionary for MongoDB query.
    4. Filters out labels based on the master dictionary.

    Args:
        db: The database connection object.
        search_query_params_ids_dict: Dictionary containing search query parameters as keys and list of IDs as values.

    Returns:
        Tuple containing:
        - Dictionary of filtered labels based on search query parameters.
        - List of unique and flattened labels based on search query parameters.
        - Dictionary of master labels for each search query parameter.
    """

    # Retrieve filtered labels from database based on the search query parameters
    search_query_params_filtered_labels = get_labels_from_ids(
        db, 'positive_labels', search_query_params_ids_dict)

    # Flatten and find unique labels from the filtered labels
    flattened_unique_labels_search_query_params = [
        item for sublist in search_query_params_filtered_labels.values() for item in sublist]

    # Create ObjectIds for MongoDB query based on the search query parameters
    print("Type of search_query_params_ids_dict:",
          type(search_query_params_ids_dict))
    print("Value of search_query_params_ids_dict:", search_query_params_ids_dict)
    # Create ObjectIds for MongoDB query based on the search query parameters
    if isinstance(search_query_params_ids_dict, list):
        object_ids_search_query_params = [ObjectId(
            id) for single_dict in search_query_params_ids_dict for id_list in single_dict.values() for id in id_list]
    elif isinstance(search_query_params_ids_dict, dict):
        object_ids_search_query_params = [ObjectId(
            id) for id_list in search_query_params_ids_dict.values() for id in id_list]
    else:
        raise TypeError(
            "search_query_params_ids_dict must be either a dictionary or a list of dictionaries")

    # Create master dictionary for MongoDB query
    master_label_dict_label_search_query_params = create_label_master_dict(
        db, 'positive_labels', {"_id": {"$in": object_ids_search_query_params}})

    # Obtain unique labels from the search query parameters
    unique_labels_search_query_params = get_unique_labels(
        master_label_dict_label_search_query_params, search_query_params_filtered_labels)

    # Filter out labels based on the master dictionary
    search_query_param_labels_in_master_label_dict = {}
    for category, labels in unique_labels_search_query_params.items():
        search_query_param_labels_in_master_label_dict[category] = [
            label for label in labels if label in master_label_dict_label_search_query_params.values()]

    return search_query_param_labels_in_master_label_dict


def perform_fuzzy_matching_for_attributes(search_query_dict, filtered_labels, master_label_dict, threshold=90):
    print("Entered perform_fuzzy_matching_for_attributes function")  # Debug 1

    results = {}
    all_matched_labels = []

    def process_single_dict(search_query_dict):
        # Debug 2
        print(f"Entered process_single_dict with {search_query_dict}")
        nonlocal results
        nonlocal all_matched_labels
        for attribute, value in search_query_dict.items():
            # Debug 3
            print(f"Processing attribute: {attribute}, value: {value}")

            if not value:
                continue

            if not isinstance(value, list):
                value = [value]

            label_list_fuzzy_matched_with_attribute = get_fuzzy_scores(
                value, filtered_labels, min_score=threshold)
            # Debug 4
            print(
                f"Matched labels for attribute {attribute}: {label_list_fuzzy_matched_with_attribute}")

            label_list_fuzzy_matched_with_attribute = [
                label for label, score in label_list_fuzzy_matched_with_attribute.items() if score > threshold]

            if attribute in results:
                results[attribute].extend(
                    label_list_fuzzy_matched_with_attribute)
            else:
                results[attribute] = label_list_fuzzy_matched_with_attribute

            all_matched_labels.extend(label_list_fuzzy_matched_with_attribute)

    if isinstance(search_query_dict, dict):
        process_single_dict(search_query_dict)
    elif isinstance(search_query_dict, list) and all(isinstance(d, dict) for d in search_query_dict):
        for single_dict in search_query_dict:
            process_single_dict(single_dict)
    else:
        raise TypeError(
            "Invalid type for search_query_input. Must be a dictionary or a list of dictionaries.")

    results = {k: list(set(v)) for k, v in results.items()}
    all_matched_labels = list(set(all_matched_labels))
    print(f"Final all_matched_labels: {all_matched_labels}")  # Debug 5

    _, label_list_most_relevant_to_search_concerns_master_labels = get_mapped_labels(
        all_matched_labels, master_label_dict)

    return results, all_matched_labels, label_list_most_relevant_to_search_concerns_master_labels


def get_weighted_unique_list(search_query_master_label_list,
                             search_query_param_labels_in_master_label_dict,
                             search_query_params_fuzzy_master_labels):

    # The set ensures uniqueness
    seen = set()

    # Prioritized order
    final_list = []

    # Process search_query_master_label_dict first
    for item in search_query_master_label_list:
        if item not in seen:
            seen.add(item)
            final_list.append(item)

    # Extracting values from the dictionary
    for key in search_query_param_labels_in_master_label_dict:
        for item in search_query_param_labels_in_master_label_dict[key]:
            if item not in seen:
                seen.add(item)
                final_list.append(item)

    # Finally, process search_query_params_fuzzy_master_labels
    for item in search_query_params_fuzzy_master_labels:
        if item not in seen:
            seen.add(item)
            final_list.append(item)

    return final_list


def final_search_query_parameters(search_query, search_query_param_master_labels, search_query_master_labels_weights, master_label_dict, search_query_master_labels, labels, label_count=5):
    """
    Extracts and returns relevant search query parameters using AI.

    Parameters:
    - search_query (str): The search query string.
    - search_query_param_master_labels (list): Master labels specifically for search query parameterization.
    - search_query_master_labels_weights (list): List of master labels with their respective weights.
    - master_label_dict (dict): Dictionary mapping master labels to original labels.
    - search_query_master_labels (list): List of master labels extracted from search queries.
    - labels (list): Labels for skincare types.
    - label_count (int): Number of labels to return, defaults to 5.

    Returns:
    tuple: Contains the following 3 items.
    - master_labels_extracted_using_AI (list): Master labels extracted using AI from the final list.
    - final_labels_extracted_using_AI_weights (list): Extracted labels with normalized weights.
    - original_labels_search_dict_using_AI (dict): Dictionary of original labels based on extracted master labels.

    """

    # Use openAI to extract the master labels from the final list
    master_labels_extracted_using_AI = extract_master_labels_using_openAI(
        search_query, search_query_param_master_labels, label_count)

    # Filter labels to those extracted using AI and their corresponding weights
    final_labels_extracted_using_AI_weights = [
        item for item in search_query_master_labels_weights if item[0] in master_labels_extracted_using_AI]

    # Normalize weights of the final labels
    total_weight = sum([weight[2]
                       for weight in final_labels_extracted_using_AI_weights])
    for i, item in enumerate(final_labels_extracted_using_AI_weights):
        label, count, weight = item
        normalized_weight = weight / total_weight
        final_labels_extracted_using_AI_weights[i] = (
            label, count, normalized_weight)

    # Map the final labels to their original labels
    original_labels_search_dict_using_AI = get_original_labels_for_master_labels(
        final_labels_extracted_using_AI_weights, master_label_dict)

    return master_labels_extracted_using_AI, final_labels_extracted_using_AI_weights, original_labels_search_dict_using_AI


############################ Retreival and re-ranking of products from MongoDB ##########################################


# Initialize Variables:
def initialize():
    labels_in_doc = []
    total_label_counts = defaultdict(int)
    product_data_with_relevance = []
    return labels_in_doc, total_label_counts, product_data_with_relevance


def extract_id_and_document(product_metadata, docs_dict):
    id = product_metadata.get('metadata', {}).get('_id', None)

    # Skip the entry if the id is None
    if id is None:
        return None, None

    doc = docs_dict.get(id, None)

    # Skip the entry if the doc is None (i.e., id is not found in docs_dict)
    if doc is None:
        return None, None

    return id, doc


def get_skin_type_overviews(doc):
    skin_type_overviews = {}
    for item in doc['skin_type']:
        name = item.get('name', 'N/A')
        positive_labels_aggregated = {}
        for label_item in item.get('positive_labels', []):
            label_name = label_item.get('label', 'N/A')
            reviews_count = label_item.get('reviews', 0)
            positive_labels_aggregated[label_name] = reviews_count
        skin_type_overviews[name] = positive_labels_aggregated
    return skin_type_overviews


def calculate_master_label_counts(positive_labels_in_doc, master_label_dict, total_label_counts=None):
    master_label_counts, master_label_counts_str = get_master_label_counts(
        positive_labels_in_doc, master_label_dict)

    if total_label_counts is not None:
        for label, count in master_label_counts.items():
            total_label_counts[label] += count
        return master_label_counts, total_label_counts, master_label_counts_str
    else:
        return master_label_counts, master_label_counts_str


def get_product_review_overviews(doc, master_label_dict):
    skin_type_overview = [item for item in doc['skin_type']
                          if item.get('name') == 'overview']
    positive_labels_list = skin_type_overview[0].get(
        'positive_labels', []) if skin_type_overview else []

    # Convert list of dictionaries to dictionary of dictionaries
    positive_labels_in_doc = {item['label']: item for item in positive_labels_list}

    overview_str = " • ".join(
        [f"{label} ({details['reviews']})" for label, details in positive_labels_in_doc.items()])
    master_label_counts, master_label_counts_str = calculate_master_label_counts(
        positive_labels_in_doc, master_label_dict)
    sorted_positive_labels_in_doc = dict(sorted(
        positive_labels_in_doc.items(), key=lambda item: item[1]['reviews'], reverse=True))

    return positive_labels_in_doc, overview_str, master_label_counts, master_label_counts_str, sorted_positive_labels_in_doc


def labels_fuzzy_after_cosine(search_query, id_label_list, positive_labels_in_doc):
    lower_label_set = {label.lower() for _, label in id_label_list}
    sorted_labels = sorted((label_dict for label_dict in positive_labels_in_doc if label_dict['label'].lower(
    ) in lower_label_set), key=lambda x: x['reviews'], reverse=True)
    cosine_id_label_list_to_product_labels = [
        label_dict['label'] for label_dict in sorted_labels]
    fuzzy_scores_after_cosine_matching_available_cosine_id_label_list = get_fuzzy_scores(
        search_query, cosine_id_label_list_to_product_labels)
    fuzzy_scores_after_cosine_matching_available_cosine_id_label_list = [
        label for label, score in fuzzy_scores_after_cosine_matching_available_cosine_id_label_list.items() if score > 80]
    return cosine_id_label_list_to_product_labels, fuzzy_scores_after_cosine_matching_available_cosine_id_label_list


def get_relevance_score_with_boost(base_relevance_score, total_boost_v2):
    return base_relevance_score + total_boost_v2


def process_product(product_metadata, docs_dict, final_labels_extracted_using_AI, search_query_dict_final, master_label_dict, top_3_reviews=None):

    _, doc = extract_id_and_document(product_metadata, docs_dict)
    print(f"doc after extract_id_and_document:{doc}")

    if not doc:
        return None

    # Code is mainly using data from MongoDB collection product_review_summaries

    def get_total_product_reviews(doc):
        reviews_count = doc.get('total_reviews', 0)
        return reviews_count

    def get_overview_labels(doc, master_label_dict):
        skin_type_overviews = get_skin_type_overviews(doc)
        positive_labels_in_doc, overview_str, master_label_counts, master_label_counts_str, sorted_positive_labels_in_doc = get_product_review_overviews(
            doc, master_label_dict)
        return skin_type_overviews, overview_str, positive_labels_in_doc, master_label_counts, master_label_counts_str, sorted_positive_labels_in_doc

    def get_AI_extracted_labels(master_label_counts, final_labels_extracted_using_AI):
        filtered_master_label_counts_using_AI_dict, filtered_master_label_counts_using_AI_dict_str = filter_master_label_counts_vs_AI_search_weights(
            master_label_counts, final_labels_extracted_using_AI)
        return filtered_master_label_counts_using_AI_dict, filtered_master_label_counts_using_AI_dict_str

    def calculate_relevance_scores(sorted_positive_labels_in_doc, reviews_count, filtered_master_label_counts_using_AI_dict):
        base_relevance_score = calculate_base_relevance_score(
            sorted_positive_labels_in_doc, reviews_count)
        total_boost_v2 = calculate_total_boost(
            filtered_master_label_counts_using_AI_dict, reviews_count, final_labels_extracted_using_AI)
        relevance_score_with_boost = get_relevance_score_with_boost(
            base_relevance_score, total_boost_v2)
        return relevance_score_with_boost

    reviews_count = get_total_product_reviews(doc)
    skin_type_overviews, overview_str, positive_labels_in_doc, master_label_counts, master_label_counts_str, sorted_positive_labels_in_doc = get_overview_labels(
        doc, master_label_dict)
    filtered_master_label_counts_using_AI_dict, filtered_master_label_counts_using_AI_dict_str = get_AI_extracted_labels(
        master_label_counts, final_labels_extracted_using_AI)
    relevance_score_with_boost = calculate_relevance_scores(
        sorted_positive_labels_in_doc, reviews_count, filtered_master_label_counts_using_AI_dict)
    title_boost = TitleBooster.get_title_boost(
        product_metadata, search_query_dict_final)
    print(f'Title Boost: {title_boost}')
    category_boost = CategoryBooster.get_category_boost(
        product_metadata, search_query_dict_final)
    print(f'Category Boost: {category_boost}')
    final_relevance_score = relevance_score_with_boost + title_boost + category_boost

    if top_3_reviews is None:
        top_3_reviews = "top 3 reviews have not been searched for or specified"

    return {
        'product_metadata': product_metadata,  # dictionary of dictionaries
        'reviews_count': reviews_count,  # float
        'top_3_reviews': top_3_reviews,  # dictionary of dictionaries
        'relevance_score': final_relevance_score,  # float
        'skin_type_overviews': skin_type_overviews,  # dictionary of dictionaries
        'overview_str': overview_str,  # string
        'positive_labels_in_doc': positive_labels_in_doc,  # list of dictionaries
        'master_label_counts': master_label_counts,  # string
        'master_label_counts_str': master_label_counts_str,  # string
        'filtered_master_label_counts_using_AI_dict_str': filtered_master_label_counts_using_AI_dict_str  # string
    }


def rerank_products(product_data_with_relevance):
    return sorted(product_data_with_relevance, key=lambda x: x['relevance_score'], reverse=True)


############################ Cleanup and displaying of data ##########################################

def format_date_difference(start_date, end_date):
    """Format the difference between two dates in terms of days, months, or years."""
    delta = end_date - start_date
    days = delta.days

    if days < 30:
        return f"{days} days ago"
    elif days < 365:
        months = days // 30
        return f"{months} month{'s' if months > 1 else ''} ago"
    else:
        years = days // 365
        return f"{years} year{'s' if years > 1 else ''} ago"


def extract_sorted_labels(data_str, max_labels=5):
    labels = data_str.split(' • ')
    label_counts = [(label, int(label.split(' ')[-1].replace('(', '').replace(')', '')))
                    for label in labels if label.split(' ')[-1].replace('(', '').replace(')', '').isdigit()]
    return ', '.join(label for label, _ in sorted(label_counts, key=lambda x: x[1], reverse=True)[:max_labels])


def get_match_ratio(concern, review_text):
    return fuzz.token_set_ratio(concern.lower(), review_text)


def extract_product_data(metadata):
    # Check if metadata is a tuple
    if isinstance(metadata, tuple):
        # Access the first element (the actual metadata)
        metadata = metadata[0]

    # Check if metadata contains the 'metadata' key
    if 'metadata' in metadata:
        metadata = metadata['metadata']

    return {
        'product_id': metadata.get('_id', 'N/A'),
        'brand': metadata.get('brand', 'N/A'),
        'title': metadata.get('title', 'N/A'),
        'master_category': metadata.get('master_category', 'N/A'),
        'refined_category': metadata.get('refined_category', 'N/A'),
        'price': metadata.get('price', 'N/A')
    }


def process_reviews(reviews_dict, search_query_dict_final):
    def process_single_dict(single_dict, reviews):
        local_matched_skin_concerns = []
        local_reviews_with_match_ratios = []

        for _, review in reviews.items():
            review_text = review['desc']
            match_ratio = 0

            for concern in single_dict.get('skin_concern', []):
                current_ratio = get_match_ratio(concern, review_text)
                if current_ratio > 80:
                    if current_ratio > match_ratio:
                        match_ratio = current_ratio
                        local_matched_skin_concerns.append(concern)

            local_reviews_with_match_ratios.append((review, match_ratio))

        return local_matched_skin_concerns, sorted(local_reviews_with_match_ratios, key=lambda x: x[1], reverse=True)

    matched_skin_concerns = []
    reviews_with_match_ratios = []

    # Extract nested reviews from reviews_dict
    top_3_reviews = reviews_dict.get('reviews', {})

    if isinstance(search_query_dict_final, dict):
        matched_skin_concerns, reviews_with_match_ratios = process_single_dict(
            search_query_dict_final, top_3_reviews)
    elif isinstance(search_query_dict_final, list) and all(isinstance(d, dict) for d in search_query_dict_final):
        for single_dict in search_query_dict_final:
            local_matched_skin_concerns, local_reviews_with_match_ratios = process_single_dict(
                single_dict, top_3_reviews)
            matched_skin_concerns.extend(local_matched_skin_concerns)
            reviews_with_match_ratios.extend(local_reviews_with_match_ratios)

        # Sort again if we had multiple dictionaries
        reviews_with_match_ratios = sorted(
            reviews_with_match_ratios, key=lambda x: x[1], reverse=True)
    else:
        print("Invalid input type.")

    return matched_skin_concerns, reviews_with_match_ratios


def print_review(idx, review):
    metadata = review['metadata']
    print(
        f"Review {idx+1} (Score: {review['custom_score']}, Matched Data: {review['matched_data']}):")
    print(f"Title: {metadata.get('title', 'N/A')}")
    print(f"Description: {metadata.get('desc', 'N/A')}")
    print(f"Name: {metadata.get('name', 'N/A')} | Age: {metadata.get('age', 'N/A')} | Country: {metadata.get('country', 'N/A')}")
    print(f"Skin Concern: {metadata.get('skin_concern', 'N/A')} | Skin Tone: {metadata.get('skin_tone', 'N/A')} | Skin Type: {metadata.get('skin_type', 'N/A')}")
    print(f"Rating: {metadata.get('rating', 'N/A')} | Promoted: {metadata.get('promoted', 'N/A')}\nPosted: {metadata.get('created_at', 'N/A')}")

# Check if 'created_at' is a string or a datetime.datetime object
    created_at = metadata.get('created_at', 'N/A')
    if isinstance(created_at, str):
        created_at_date = datetime.strptime(
            created_at, "%Y-%m-%d").date()  # Adjust the format if needed
    elif isinstance(created_at, datetime):
        created_at_date = created_at.date()
    else:
        created_at_date = None

    if created_at_date:
        print(
            f"Posted: {format_date_difference(created_at_date, date.today())}\n")
    else:
        print(f"Posted: N/A\n")


def print_product_data(data):
    print(
        f"Original relevance score: {data['relevance_score'] - (0.1 * len(set(data['matched_skin_concerns'])) + (1 if data['matched_skin_concerns'] else 0))}")
    print(f"Updated relevance score: {data['relevance_score']}")
    print(f"Product ID: {data['product_id']}")
    # Inserted reviews_count here
    print(
        f"Product: {data['brand']} {data['title']} (Reviews: {data['reviews_count']})")
    print(
        f"Category: {data['refined_category']}, {data['master_category']} | Price: ${data['price']}")
    print(
        f"Matched product labels to search query: {data['filtered_master_label_counts_using_AI_dict_str']}")
    print(f"Matched skin concern to review: {data['matched_skin_concerns']}")
    print(f"Top Features: {data['top_5_master_labels']}")
    print(f"Most Relevant Reviews:")

    for idx, (review, _) in enumerate(data['sorted_top_3_reviews']):
        print_review(idx, review)
    print("\n====================\n")
