# Class of functions which outline boosting metadata fields which match with a search query dictionary
from fuzzywuzzy import fuzz
from ...helpers.skincare_labels import ingredients_dictionary

# Download nltk punkt tokenizer if not already downloaded
try:
    from nltk.tokenize import word_tokenize
except LookupError:
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import word_tokenize


DEBUG_MODE = True  # Global debug flag. Set to False to suppress debug outputs.


def debug_print(message):
    if DEBUG_MODE:
        print(f"[Debug] {message}")


class SimilarityCalculator:
    @staticmethod
    def token_set_ratio(text1, text2, threshold=60):
        debug_print(f"Comparing '{text1}' and '{text2}' using token_set_ratio")
        debug_print(f"--- Token Set Ratio ---")
        similarity = fuzz.token_set_ratio(text1, text2)
        debug_print(f"Comparing '{text1}' and '{text2}': {similarity}")
        return similarity if similarity > threshold else 0

    @staticmethod
    def exact_match_boost(query_value, metadata_value):
        print(
            f"Comparing '{query_value}' and '{metadata_value}' using exact_match")
        query_value, metadata_value = query_value.lower(
        ), metadata_value.lower()  # Case-insensitive
        return 100 if query_value in metadata_value.split() else 0

    @staticmethod
    def partial_match_boost(query_value, metadata_value):
        print(
            f"Comparing '{query_value}' and '{metadata_value}' using partial_match")
        query_value, metadata_value = query_value.lower(
        ), metadata_value.lower()  # Case-insensitive
        for query_word in query_value.split():
            if any(query_word in metadata_word for metadata_word in metadata_value.split()):
                return 100
        return 0

    @staticmethod
    def synonym_boost(query_value, metadata_value):
        query_value = query_value.lower()
        metadata_value = metadata_value.lower()
        metadata_words = set(metadata_value.split())
        query_words = set(query_value.split())

        # Loop through each word in metadata_words
        for metadata_word in metadata_words:
            # If the word is a key in ingredients_dictionary
            if metadata_word in ingredients_dictionary:
                if metadata_word in query_words:
                    print(f"Debug: Found key {metadata_word} in query")
                    return 100
                for synonym in ingredients_dictionary[metadata_word]:
                    if synonym.lower() in query_words:
                        print(
                            f"Debug: Found synonym {synonym.lower()} for key {metadata_word} in query")
                        return 100
            # If the word is a synonym in ingredients_dictionary
            for key, synonyms in ingredients_dictionary.items():
                if metadata_word in [synonym.lower() for synonym in synonyms]:
                    if key.lower() in query_words or any(synonym.lower() in query_words for synonym in synonyms):
                        print(
                            f"Debug: Found key {key.lower()} or synonym {metadata_word} matching query")
                        return 100

        print("Debug: No match found.")
        return 0

    @staticmethod
    def _get_tokenized_boost(category1, category2, boost_factor):
        try:
            tokens1 = set(word_tokenize(category1.lower()))
            tokens2 = set(word_tokenize(category2.lower()))
        except LookupError:
            import nltk
            nltk.download('punkt')
            tokens1 = set(word_tokenize(category1.lower()))
            tokens2 = set(word_tokenize(category2.lower()))

        # Calculate the boost based on the intersection of tokens
        common_tokens = tokens1.intersection(tokens2)
        total_tokens = len(tokens1) + len(tokens2)
        if total_tokens == 0:
            return 0
        token_boost = len(common_tokens) / total_tokens
        print(f"Tokenized Boost: {token_boost * boost_factor}")  # Debug print
        return token_boost * boost_factor


class Booster:
    @staticmethod
    def _get_boost(metadata_value, query_value, boost_factor, method='token_set_ratio'):
        debug_print(f"===== GET BOOST START =====")
        debug_print(
            f"metadata_value={metadata_value}, type={type(metadata_value)}")
        similarity = 0
        try:
            if method == 'token_set_ratio':
                similarity = SimilarityCalculator.token_set_ratio(
                    metadata_value, query_value)
            elif method == 'exact_match':
                similarity = SimilarityCalculator.exact_match_boost(
                    metadata_value, query_value)
            elif method == 'partial_match':
                similarity = SimilarityCalculator.partial_match_boost(
                    metadata_value, query_value)
            elif method == 'synonym':
                similarity = SimilarityCalculator.synonym_boost(
                    metadata_value, query_value)
            else:
                raise ValueError(f'Unknown method: {method}')
        except Exception as e:
            print(f"Error in _get_boost: {e}")
            return 0

        boost = boost_factor * (similarity / 100)
        debug_print(f"Calculated boost: {boost}")
        debug_print(f"===== GET BOOST END =====\n")
        return boost if similarity > 0 else 0


class TitleBooster(Booster):
    @staticmethod
    def get_title_boost(product_metadata, search_query_dict):
        def process_single_dict(single_dict):
            title_boost = 0

            debug_print(f"Processing dictionary: {single_dict}")

            # For 'ingredients'
            if 'ingredients' in single_dict:
                ingredients = single_dict['ingredients']
                debug_print(f"Comparing ingredients: {ingredients}")
                if isinstance(ingredients, str):
                    ingredients = [ingredients]
                for ingredient in ingredients:
                    for method, weight in weight_factors.items():
                        boost = Booster._get_boost(
                            product_title, ingredient, 1, method=method)
                        title_boost += boost * weight

            # Your existing logic for 'product_category'
            categories = single_dict.get('product_category', [])
            if isinstance(categories, str):
                categories = [categories]
            for category in categories:
                for method, weight in weight_factors.items():
                    boost = Booster._get_boost(
                        product_title, category, 1, method=method)
                    title_boost += boost * weight

            # Normalize if surpassing max limit
            if title_boost > max_boost:
                title_boost = max_boost

            return title_boost

        debug_print("\n========================")
        debug_print("Title Boost Calculation:")
        debug_print("========================")

        product_title = product_metadata.get(
            'metadata', {}).get('title', 'Not found').strip()
        max_boost = 5.0  # Maximum allowable boost

        weight_factors = {
            'exact_match': 0.4,
            'partial_match': 0.3,
            'token_set_ratio': 0.2,
            'synonym': 0.5  # This will have more weight
        }

        if isinstance(search_query_dict, dict):
            return process_single_dict(search_query_dict)
        elif isinstance(search_query_dict, list) and all(isinstance(d, dict) for d in search_query_dict):
            title_boosts = [process_single_dict(d) for d in search_query_dict]
            # Here, I'm averaging the boosts; you can decide how to aggregate them
            return sum(title_boosts) / len(title_boosts)
        else:
            print("Invalid input type.")
            return 0


class CategoryBooster(Booster):
    @staticmethod
    def get_category_boost(product_metadata, search_query_dict, master_category_boost_factor=10, refined_category_boost_factor=10):
        debug_print("\n==========================")
        debug_print("Category Boost Calculation:")
        debug_print("==========================")

        master_category = product_metadata.get('metadata', {}).get(
            'master_category', 'Not found').strip().lower()
        refined_category = product_metadata.get('metadata', {}).get(
            'refined_category', 'Not found').strip().lower()
        title = product_metadata.get('metadata', {}).get(
            'title', 'Not found').strip().lower()
        debug_print(
            f"Master Category: {master_category}, Refined Category: {refined_category}")

        category_boost = 0

        # Define a helper function to handle a single search query dictionary
        def handle_single_dict(single_query_dict):
            nonlocal category_boost
            weight_factors = {
                'exact_match': 0.5,
                'partial_match': 0.2,
                'token_set_ratio': 0.1,
                'synonym': 0.1,
                'tokenized': 0.1
            }

            categories = single_query_dict.get('product_category', [])
            if isinstance(categories, str):
                categories = [categories.lower()]

            for category in categories:
                master_category_boost = 0
                refined_category_boost = 0

                for method, weight in weight_factors.items():
                    if method == 'tokenized':
                        boost = SimilarityCalculator._get_tokenized_boost(
                            master_category, category, master_category_boost_factor)
                    else:
                        boost = Booster._get_boost(
                            master_category, category, master_category_boost_factor, method=method)
                    master_category_boost += boost * weight

                for method, weight in weight_factors.items():
                    if method == 'tokenized':
                        boost = SimilarityCalculator._get_tokenized_boost(
                            refined_category, category, refined_category_boost_factor)
                    else:
                        boost = Booster._get_boost(
                            refined_category, category, refined_category_boost_factor, method=method)
                    refined_category_boost += boost * weight

                category_boost += master_category_boost + refined_category_boost

        # Check if search_query_dict is a list or a single dictionary
        if isinstance(search_query_dict, list):
            for single_query_dict in search_query_dict:
                handle_single_dict(single_query_dict)
            # Optionally, normalize the final category_boost by dividing by the number of dictionaries
            category_boost /= len(search_query_dict)
        else:
            handle_single_dict(search_query_dict)

        return category_boost
