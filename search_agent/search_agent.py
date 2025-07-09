from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.agents import AgentExecutor, Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool

from .tools.metadata_filter_tool import MetadataFilterTool
from .tools.product_search_tool import ProductSearchTool
from .tools.review_search_tool import ReviewSearchTool
from .tools.product_info_tool import ProductInfoTool


class SearchAgent():
    """
    A specialized conversational agent designed for product-related inquiries.

    This agent leverages a suite of tools to search for products, retrieve product
    information, filter by metadata, and find reviews. It is structured as a
    LangChain AgentExecutor to process user queries.
    """
    def __init__(self):
        """
        Initializes the SearchAgent.

        This constructor sets up the language model, initializes the available tools,
        defines the system message prompt, and builds the final AgentExecutor
        that drives the agent's logic.
        """
        # Set OpenAI LLM and embeddings
        llm_chat = self._set_llm()

        tools = self._set_tools()

        system_message = SystemMessage(
            content=(
                "Do your best to answer the questions. "
                "Feel free to use any tools available to look up relevant information, only if necessary. "
                "Do not invent information."
            )
        )

        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message
        )
        agent = OpenAIFunctionsAgent(llm=llm_chat, tools=tools, prompt=prompt)
        self.search_agent = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            max_iterations=3,
            verbose=True,
        )

    def run_agent(self, input: str):
        """
        Executes the search agent with a given user input.

        Args:
            input (dict): A dictionary containing the user's query and optional
                          product IDs. Expected format:
                          {'query': str, 'product_ids': Optional[list[str]]}

        Returns:
            str: The agent's response to the input query.
        """
        agent_input = ''

        user_input = input['query']

        if 'product_ids' in input and len(input['product_ids']) > 0:
            agent_input = f'Query: {user_input}, product ids: {input["product_ids"]}'
        else:
            agent_input = f'{user_input}'

        print('\n\nagent_input:', agent_input)
        return self.search_agent.run(agent_input)

    def use_as_tool(self):
        """
        Wraps the agent's run method into a LangChain Tool.

        This allows the entire SearchAgent to be used as a single, powerful tool
        by another, higher-level agent.

        Returns:
            Tool: A LangChain Tool instance.
        """
        return Tool(
            name="search_product_review",
            func=self.run_agent,
            description="""If the user is asking to search for a product or review. Input: {"query": A well formulated question, "product_ids": A list of prducts ids related to the query}""",
        )

    def _set_tools(self):
        """
        Initializes and returns the list of tools available to the agent.

        Returns:
            list[Tool]: A list of tools for product search, review search,
                        metadata filtering, and product information retrieval.
        """
        return [ProductSearchTool().get_tool(), ReviewSearchTool().get_tool(), MetadataFilterTool().get_tool(), ProductInfoTool().get_tool()]

    def _set_llm(self):
        """
        Configures and returns the language model for the agent.

        Returns:
            ChatOpenAI: An instance of the ChatOpenAI model.
        """
        return ChatOpenAI(temperature=0.9, model='gpt-3.5-turbo-0613', client='')
