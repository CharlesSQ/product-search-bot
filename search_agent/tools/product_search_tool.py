import os
import pinecone

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.agents import Tool

from dotenv import load_dotenv
from .chains.retrieval_reduce_qa import create_retrieval_reduce_qa_tool

load_dotenv()

# Access environment variables
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENVIROMENT = os.environ['PINECONE_ENVIROMENT']
PINECONE_INDEX_NAME = os.environ['PINECONE_INDEX_NAME']


class ProductSearchTool():
    """
    A tool for performing general semantic searches for products.

    This tool uses a custom retrieval and question-answering chain (`retrieval_reduce_qa`)
    to find relevant products in a Pinecone vector store and then summarize the
    results according to a predefined template.
    """
    def __init__(self):
        """
        Initializes the ProductSearchTool.

        This constructor sets up the connection to Pinecone, initializes the OpenAI
        embeddings and language model, and creates the retrieval QA chain with
        specific templates for summarizing product information.
        """

        # Set openain models
        embeddings = OpenAIEmbeddings(client='')
        llm_chat = ChatOpenAI(
            temperature=0.9, model='gpt-3.5-turbo-0613', client='')

        # Set Pinecone vectorstore
        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIROMENT
        )

        vectorstore = Pinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME, embedding=embeddings, text_key='title', namespace='products')

        # Create retrieval reduce qa chain
        reduce_template = """Write a concise summary of the following:
        "{content}"
        Return the product name, product_id, brand, price, details and link.
        CONCISE SUMMARY:"""

        input_variables = ["page_content", "_id", "brand",
                           "price", "details", "img", "prod_link"]
        document_template = "product: {page_content}, product_id: {_id}, brand: {brand}, price: {price}, details: {details}, img: {img}, prod_link: {prod_link}"

        self.retriever = create_retrieval_reduce_qa_tool(
            llm_chat, vectorstore.as_retriever(search_kwargs={'k': 6}), reduce_template, document_template, input_variables)

    def get_tool(self):
        """
        Creates and returns a LangChain Tool instance for this retriever.

        This method wraps the retrieval QA chain into a standardized Tool object
        that can be used directly by a LangChain agent.

        Returns:
            Tool: A LangChain Tool configured for general product searches.
        """
        return Tool(
            name="search_products",
            description="Searches and returns information regarding products in the database.",
            func=self.retriever.run,
            coroutine=self.retriever.arun,
            return_direct=True
        )
