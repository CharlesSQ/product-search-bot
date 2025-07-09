import pinecone
import os

from .custom_self_query_retriever.base import CustomSelfQueryRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.agents import Tool

from dotenv import load_dotenv

load_dotenv()

# Access environment variables
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENVIROMENT = os.environ['PINECONE_ENVIROMENT']
PINECONE_INDEX_NAME = os.environ['PINECONE_INDEX_NAME']


metadata_field_info = {}

metadata_field_info["prod_id"] = {
    "description": "The product id",
    "type": "string"
}


class ReviewSearchTool():
    """
    A tool for retrieving product reviews based on product IDs.

    This tool uses a self-querying retriever to fetch reviews from a Pinecone
    vector store by filtering on the `prod_id` metadata field in the 'reviews'
    namespace.
    """
    def __init__(self):
        """
        Initializes the ReviewSearchTool.

        This constructor sets up the connection to Pinecone, initializes the OpenAI
        embeddings and language model, and configures the self-querying retriever
        to specifically look for product reviews by ID.
        """

        llm_chat = ChatOpenAI(
            temperature=0.9, model='gpt-3.5-turbo-0613', client='')

        embeddings = OpenAIEmbeddings(client='')

        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIROMENT
        )

        vectorstore = Pinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME, embedding=embeddings, text_key='title', namespace='reviews')

        document_content_description = "Reviews data"

        retry_message = "No results found, ask user to be more specific. Never mention the product id or a reference to it."

        self.retriever = CustomSelfQueryRetriever.from_llm(
            llm_chat, vectorstore, document_content_description, metadata_field_info, max_retry=3, retry_message=retry_message)

    def retriever_tool(self, input):
        """
        Synchronously executes the retriever to find product reviews by ID(s).

        Args:
            input (str or list[str]): A single product ID or a list of product IDs.

        Returns:
            list[Document]: A list of LangChain documents containing the reviews for
                          the requested products.
        """
        print('input:', input)
        k = 5

        if isinstance(input, list):
            product_ids = ''
            for prod_id in input:
                product_ids += prod_id + ', '

            query_input = f'Reviews of products with ids {product_ids}, return {k} reviews. Use the eq comparator.'

        elif isinstance(input, str):
            query_input = f'Reviews of product with id {input}, return {k} reviews. Use the eq comparator.'

        return self.retriever.get_relevant_documents(query=query_input)

    def aretriever_tool(self, input: str):
        """
        Asynchronously executes the retriever to find product reviews by ID(s).

        Args:
            input (str or list[str]): A single product ID or a list of product IDs.

        Returns:
            list[Document]: A list of LangChain documents containing the reviews for
                          the requested products.
        """
        k = 5

        if isinstance(input, list):
            product_ids = ''
            for prod_id in input:
                product_ids += prod_id + ', '

            query_input = f'Reviews of products with ids {product_ids}, return {k} reviews. Use the eq comparator.'

        elif isinstance(input, str):
            query_input = f'Reviews of product with id {input}, return {k} reviews. Use the eq comparator.'

        return self.retriever.aget_relevant_documents(query=query_input)

    def get_tool(self):
        """
        Creates and returns a LangChain Tool instance for this retriever.

        This method wraps the retriever logic into a standardized Tool object
        that can be used by a LangChain agent for review lookups.

        Returns:
            Tool: A LangChain Tool configured for product review retrieval by ID.
        """
        return Tool(
            name="search_reviews",
            description="Searches and returns product reviews from the database. Input: {'prod_ids': An array of product ids}",
            func=self.retriever_tool,
            coroutine=self.aretriever_tool
        )
