from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from conversational_agent import ConversationalAgent

import time

llm_chat = ChatOpenAI(temperature=0.9, model='gpt-3.5-turbo-0613', client='')

# Set conversation memory buffer
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5)

agent = ConversationalAgent(
    llm_model=llm_chat, memory=memory)


def main():
    user_prompt = input("\n\nUser: ")
    response = agent.run(user_prompt)
    print('\n\nAssistant: ' + response)


if __name__ == '__main__':
    while True:
        main()
        time.sleep(1)
