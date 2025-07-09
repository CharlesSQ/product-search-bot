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

metadata_field_info["master_category"] = {
    "description": "The product category",
    "type": "string or list[string]",
}
metadata_field_info["refined_category"] = {
    "description": "The category of the product",
    "type": "string",
}
metadata_field_info["price"] = {
    "description": "The price of the product",
    "type": "tuple",
}
metadata_field_info["brand"] = {
    "description": "The brand of the product",
    "type": "string",
}
metadata_field_info["ingredients"] = {
    "description": "A description of the product ingredients",
    "type": "string"
}


class MetadataFilterTool():
    """
    A tool designed to search for products by filtering on their metadata.

    This tool utilizes a custom self-querying retriever (`CustomSelfQueryRetriever`)
    to translate a natural language query into a structured query against a Pinecone
    vector store, filtering on fields like brand, price, category, etc.
    """
    def __init__(self):
        """
        Initializes the MetadataFilterTool.

        This constructor sets up the connection to Pinecone, initializes the OpenAI
        embeddings and language model, and configures the self-querying retriever
        with the necessary metadata field information.
        """

        llm_chat = ChatOpenAI(
            temperature=0.9, model='gpt-3.5-turbo-0613', client='')

        embeddings = OpenAIEmbeddings(client='')

        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIROMENT
        )

        vectorstore = Pinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME, embedding=embeddings, text_key='title', namespace='products')

        document_content_description = "Products data"

        retry_message = "No results found, ask user to be more specific and don't mention de product id"

        self.retriever = CustomSelfQueryRetriever.from_llm(
            llm_chat, vectorstore, document_content_description, metadata_field_info, max_retry=3, retry_message=retry_message)

    def retriever_tool(self, input):
        """
        Synchronously executes the retriever tool to find relevant documents.

        Args:
            input (str): The natural language query from the user.

        Returns:
            list[Document]: A list of LangChain documents relevant to the query.
        """
        print('input:', input)
        k = 5

        query_input = f'{input}. Return {k} products.'
        return self.retriever.get_relevant_documents(query=query_input)

    def aretriever_tool(self, input: str):
        """
        Asynchronously executes the retriever tool to find relevant documents.

        Args:
            input (str): The natural language query from the user.

        Returns:
            list[Document]: A list of LangChain documents relevant to the query.
        """
        k = 5

        query_input = f'{input}. Return {k} products.'
        return self.retriever.aget_relevant_documents(query=query_input)

    def get_tool(self):
        """
        Creates and returns a LangChain Tool instance for this retriever.

        This method wraps the retriever logic into a standardized Tool object
        that can be used by a LangChain agent.

        Returns:
            Tool: A LangChain Tool configured for metadata-based product searches.
        """
        return Tool(
            name="metadata_filter",
            description="Useful for when you need to search for products based on metadata like brand, price, ingredients, master_category, refined_category. Input: {'query': 'a well formulated user input'}",
            func=self.retriever_tool,
            coroutine=self.aretriever_tool
        )
