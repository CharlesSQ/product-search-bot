from langchain.chains import LLMChain
from langchain.agents.agent import AgentExecutor
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import LLMSingleActionAgent, AgentExecutor, Tool

from .output_parser import ConversationAgentOutputParser
from .prompt_template import ConversationlAgentPromptTemplate
from .prompts import TEMPLATE_INSTRUCTIONS, SUFFIX
from search_agent import SearchAgent


class ConversationalAgent():
    """
    An agent responsible for orchestrating the conversation flow.

    This agent acts as a high-level router. It interprets the user's input,
    maintains the conversation history, and decides whether to provide a direct
    answer or to delegate the task to a specialized tool, such as the SearchAgent.
    It uses a custom prompt and output parser to manage the interaction loop.
    """

    def __init__(self, llm_model: BaseChatModel, memory: ConversationBufferWindowMemory, max_retry=1):
        """
        Initializes the ConversationalAgent.

        Args:
            llm_model (BaseChatModel): The language model to be used by the agent.
            memory (ConversationBufferWindowMemory): The memory object to store conversation history.
            max_retry (int): The maximum number of retries if the agent fails to produce a valid action.
        """
        self.llm_model = llm_model
        self.memory = memory
        self.max_retry = max_retry

        # Set up the agent
        self.tools = self._set_tools()
        prompt = self._set_up_prompt(self.tools)

        self.llm_chain = LLMChain(
            llm=self.llm_model, prompt=prompt)

    def run(self, input: str):
        """
        Executes a single turn of the conversation.

        This method takes the user's input, runs the LLM chain, parses the output,
        and determines the next step. It can either return a final answer or call a
        tool. It includes a retry loop to handle parsing errors and manages saving
        the context to memory.

        Args:
            input (str): The user's message.

        Returns:
            str: The agent's response to the user.
        """
        # Get the memory
        memory = self.memory.load_memory_variables({})

        # Run the chain while retry is less than max_retry
        retry = 0
        while retry < self.max_retry:
            output = self.llm_chain.run(
                {"input": input, "chat_history": memory['chat_history']})
            # print('chain output:', output)

            output_parser = self._set_up_ouput_parser()
            parse = output_parser.parse(output)
            # print('parse:', parse)
            # print('parse action:', parse['action'])

            # Validate the output has the key action, if present check if the value is "Final answer"
            if 'error' in parse:
                print('No action found in output')
                retry += 1

            else:
                action: str = parse['action']
                if action.lower() == 'final_answer':
                    final_output = parse['action_input']
                    self.memory.save_context({"user": input}, {
                        "ai": final_output})
                    return final_output
                else:
                    # call tool
                    print('\n\naction:', action)
                    print('\n\naction_input:', parse["action_input"])

                    # Verify if the action is a tool
                    if action in [tool.name for tool in self.tools]:
                        tool = [
                            tool for tool in self.tools if tool.name == action][0]

                        tool_output = tool.func(parse['action_input'])
                        print('\n\ntool output:', tool_output)

                        if isinstance(tool_output, dict):
                            print('\ntool output is dict')
                            output_prompt = f'Send all the params of each product from this object: {tool_output}'
                            tool_output = self.llm_model.predict(output_prompt)

                            # Clear memory
                            self.memory.clear()

                        self.memory.save_context({"user": input}, {
                            "ai": tool_output})

                        return tool_output
                    else:
                        retry += 1

    def _set_tools(self):
        """
        Initializes the tools available to the agent.

        Returns:
            list[Tool]: A list of tools, in this case containing the SearchAgent.
        """
        tools = [SearchAgent().use_as_tool()]
        return tools

    def _set_up_prompt(self, tools):
        """
        Creates and configures the prompt template for the agent.

        Args:
            tools (list[Tool]): The list of tools to be included in the prompt.

        Returns:
            ConversationlAgentPromptTemplate: The configured prompt template instance.
        """
        return ConversationlAgentPromptTemplate(
            prefix='',
            instructions=TEMPLATE_INSTRUCTIONS,
            sufix=SUFFIX,
            tools=tools,
            input_variables=["input", "chat_history"]
        )

    def _set_up_ouput_parser(self):
        """
        Initializes the output parser for the agent.

        Returns:
            ConversationAgentOutputParser: The output parser instance.
        """
        return ConversationAgentOutputParser()
