"""
This module contains the string templates for constructing the prompts used by the ConversationalAgent.

It defines the main instructions and the suffix for the prompts, which guide the
language model on how to process user input, use tools, and format its response.
"""

# An older, more complex version of the prompt instructions, detailing a step-by-step reasoning process for the agent.
TEMPLATE_INSTRUCTIONS1 = """Follow these steps to answer the user queries:

chat history: <chat-history>{chat_history}</chat-history>

Step 1 - Think if the information to answer the user input exists in "chat history".

Step 2 - If it exists go to step 3, otherwise go to step 4

Step 3 - Use the provided "chat history" to answer the user input. Respond in this Json format:
```json{{"action": "Final Answer", "action_input": "the final answer to the original user input"}}
``` 
Step 4 - Think if you need to use the tools provided below to search for the information to answer the user input:
tools:
{tools}

Step 5 - If you need to use a tool use it, otherwise go to step 6.

Step 6 - Respond in this Json format:
```json{{"action": "Final Answer", "action_input": "the final answer to the original user input"}}
```

"""

# The corresponding suffix for the older prompt template.
SUFFIX1 = """\nFollow all the steps instructions in order. Don't skip any step."""


# The main instruction template for the conversational agent.
# It instructs the agent to act as a sales assistant, use the provided context (chat history),
# and choose an appropriate action, either providing a final answer or using a tool.
# The `{tools}` placeholder is dynamically filled with the list of available tools.
TEMPLATE_INSTRUCTIONS = """You are a helpful sales assistant

You'll be provided with a "context" and a list of "actions" to answer the user.

Look for the response first in the "context", then follow the instructions below:

Actions:
- If the user input is related to the context, send this Json format:
```json{{"action": "final_answer", "action_input": "the final answer to the original user input"}}
```
{tools}


"""


# The suffix for the prompt, which includes the user's actual input.
# It also reinforces that the response must be in JSON format.
# The `{input}` placeholder is dynamically filled with the user's message.
SUFFIX = "{input}\nRespond Only in JSON format!"
