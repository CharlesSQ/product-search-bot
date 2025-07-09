from typing import List
from langchain.agents import Tool
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import SystemMessage, BaseMessage, HumanMessage, AIMessage


class UserMessage(BaseMessage):
    """Represents a message from the user in the conversation."""

    example: bool = False
    """Whether this Message is being passed in to the model as part of an example 
        conversation.
    """

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "user"

# Set up a prompt template


class ConversationlAgentPromptTemplate(BaseChatPromptTemplate):
    """
    A custom prompt template for the conversational agent.

    This class assembles the final prompt from multiple components:
    - A prefix and instructions.
    - A dynamically generated list of available tools and their descriptions.
    - The ongoing chat history.
    - A suffix containing the current user input.
    """
    # The template to use
    prefix: str
    instructions: str
    sufix: str
    tools: List[Tool]

    def _set_tool_description(self, tool_description, tool_name, tool_input):
        """
        Formats the description for a single tool, including the expected JSON output.

        Args:
            tool_description (str): The base description of the tool.
            tool_name (str): The name of the tool.
            tool_input (str): A description of the expected input for the tool.

        Returns:
            str: The fully formatted tool description.
        """
        full_description = f"""{tool_description}, send this Json format:
```json{{"action": "{tool_name}", "action_input": "{tool_input}"}}
```"""
        return full_description

    def format_messages(self, **kwargs) -> str:
        """
        Assembles the final list of messages to be sent to the LLM.

        This method takes the input variables (like chat_history and user input),
        formats the tool descriptions, and constructs a sequence of System, AI, and
        Human messages that form the complete prompt.

        Args:
            **kwargs: Arbitrary keyword arguments, expected to contain 'chat_history'
                      and other variables for formatting the prompt.

        Returns:
            list[BaseMessage]: A list of formatted messages ready for the LLM.
        """
        # Create a tools variable from the list of tools provided
        chat_context: str = "{chat_history}"

        separator = '. Input:'
        kwargs["tools"] = "\n".join(
            [f"- {self._set_tool_description(tool.description.split(separator)[0], tool.name, tool.description.split(separator)[1])}" for tool in self.tools])

        # Create a list of tool names for the tools provided
        # kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted_instructions = self.instructions.format(**kwargs)
        formates_chat_context = chat_context.format(**kwargs)
        formatted_suffix = self.sufix.format(**kwargs)
        return [SystemMessage(content=formatted_instructions), AIMessage(content=formates_chat_context), HumanMessage(content=formatted_suffix)]
