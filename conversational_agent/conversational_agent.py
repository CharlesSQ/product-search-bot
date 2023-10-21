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
    This agent is responsible for handling the conversation and route it to the search agent.
    """

    def __init__(self, llm_model: BaseChatModel, memory: ConversationBufferWindowMemory, max_retry=1):
        self.llm_model = llm_model
        self.memory = memory
        self.max_retry = max_retry

        # Set up the agent
        self.tools = self._set_tools()
        prompt = self._set_up_prompt(self.tools)

        self.llm_chain = LLMChain(
            llm=self.llm_model, prompt=prompt)

    def run(self, input: str):
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
        tools = [SearchAgent().use_as_tool()]
        return tools

    def _set_up_prompt(self, tools):
        return ConversationlAgentPromptTemplate(
            prefix='',
            instructions=TEMPLATE_INSTRUCTIONS,
            sufix=SUFFIX,
            tools=tools,
            input_variables=["input", "chat_history"]
        )

    def _set_up_ouput_parser(self):
        return ConversationAgentOutputParser()
