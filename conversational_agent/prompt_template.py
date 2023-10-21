from typing import List
from langchain.agents import Tool
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import SystemMessage, BaseMessage, HumanMessage, AIMessage


class UserMessage(BaseMessage):
    """A Message from a user."""

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
    # The template to use
    prefix: str
    instructions: str
    sufix: str
    tools: List[Tool]

    def _set_tool_description(self, tool_description, tool_name, tool_input):
        full_description = f"""{tool_description}, send this Json format:
```json{{"action": "{tool_name}", "action_input": "{tool_input}"}}
```"""
        return full_description

    def format_messages(self, **kwargs) -> str:
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
