from .openAIcompletion import get_chat_completion
import ast
from ...helpers.skincare_labels import skincare_schema


def extract_master_labels_using_openAI(search_query, master_label_list, label_count=5, max_attempts=3):
    for attempt in range(max_attempts):
        if len(search_query.split(" ")) <= 2:
            search_query = 'good for ' + search_query

        if label_count is not None:
            messages = [{"role": "user", "content": f"""Based on the skincare search query: '{search_query}', evaluate its relevance to the provided labels in this 'master_label_list':{master_label_list}'. The labels are already sorted by relevance so the order matters. Identify and include the top {label_count} labels that are directly mentioned or very strongly implied by the query, without rephrasing or altering them. Ensure that the output consists only of the exact labels from the 'master_label_list', up to the specified number of {label_count}. Labels that are not explicitly related to the query or are not present in the original 'master_label_list' should be excluded. If there are fewer than {label_count} relevant labels, include all that apply. Return the result as a Python list of strings, strictly following the given instructions, without any additional explanation or commentary."""}]
        else:
            messages = [{"role": "user", "content": f"""Based on the skincare search query: '{search_query}', evaluate its relevance to the provided labels in '{master_label_list}'. The labels are already sorted by relevance so the order matters. Identify and include labels that are directly mentioned or very strongly implied by the query, without rephrasing or altering them. Ensure that the output consists only of the exact labels from the 'master_label_list'. Labels that are not explicitly related to the query or are not present in the original 'master_label_list' should be excluded. Return the result as a Python list of strings, strictly following the given instructions, without any additional explanation or commentary."""}]

        master_label_list_extracted = get_chat_completion(messages)

        try:
            master_label_list_extracted = ast.literal_eval(
                master_label_list_extracted)
            print("Successfully extracted list directly from response")
            if master_label_list_extracted:  # If the list is not empty
                print(f"List extracted successfully in attempt {attempt + 1}")
                return master_label_list_extracted
        except (ValueError, SyntaxError):
            print(
                f"Attempt {attempt + 1} failed to extract the list, retrying...")

    print(f"All {max_attempts} attempts failed to extract the list.")
    return []


def extract_master_label_weights_using_openAI(search_query, master_label_list, label_count=5, max_attempts=3):
    # Inner function to normalize the weights
    def normalize_weights(master_labels):
        total_current_weight = sum(
            weight for label, occurrence, weight in master_labels)
        normalized_master_labels = [(label, occurrence, weight / total_current_weight)
                                    for label, occurrence, weight in master_labels]
        return normalized_master_labels

    for attempt in range(max_attempts):
        if len(search_query.split(" ")) <= 2:
            search_query = 'good for ' + search_query

        # Assume get_chat_completion(messages) is some function that you have defined elsewhere
        # to fetch the result.
        messages = [{"role": "user", "content": f"""Based on the skincare search query: '{search_query}', evaluate its relevance to the provided labels in this 'master_label_list': {master_label_list} using the 'skincare_schema': {skincare_schema}. The labels and schema properties are already sorted by relevance, so the order matters.

First, identify the aspects of the search query that correspond to the properties in the 'skincare_schema'. Assign weights to each label in 'master_label_list' according to how directly it's mentioned or implied in the search query. The weight should also consider the order of the properties in 'skincare_schema', with more weight given to properties appearing earlier in the schema.

For example, if 'skin_concern' appears first in 'skincare_schema' and is explicitly mentioned in the query, labels related to 'skin_concern' should be weighted higher. Similarly, if 'skin_type' appears next and is also mentioned in the query, labels related to 'skin_type' should be weighted next in line, but less than 'skin_concern'.

Once you've assigned weights, normalize them so that they sum up to 1. Return the result as a Python list of tuples. Each tuple should contain the exact label from 'master_label_list', its assigned weight as an integer, and its normalized weight as a float. If there are fewer than {label_count} relevant labels, include all that apply.

Return the list of tuples strictly following these instructions, without any additional explanation or commentary."""}]

        master_label_list_extracted = get_chat_completion(
            messages)  # Your external function

        try:
            master_label_list_extracted = ast.literal_eval(
                master_label_list_extracted)

            # Validate the list of tuples
            if all(isinstance(item, tuple) and len(item) == 3 for item in master_label_list_extracted):
                if all(isinstance(item[0], str) and isinstance(item[1], int) and isinstance(item[2], float) for item in master_label_list_extracted):
                    print(
                        "Successfully extracted list of tuples directly from response")

                    # Normalize the weights
                    normalized_master_labels = normalize_weights(
                        master_label_list_extracted)

                    # Make sure the list has exactly label_count number of items
                    if len(normalized_master_labels) > label_count:
                        normalized_master_labels = normalized_master_labels[:label_count]
                    elif len(normalized_master_labels) < label_count:
                        print(
                            f"Warning: Only {len(normalized_master_labels)} labels are available.")

                    return normalized_master_labels
        except (ValueError, SyntaxError):
            print(
                f"Attempt {attempt + 1} failed to extract the list, retrying...")

    print(f"All {max_attempts} attempts failed to extract the list.")
    return []
