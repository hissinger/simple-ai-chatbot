import os
import readline
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

# OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# memory key
HISTORY_MEMORY_KEY = "chat_history"


# custom input function to handle korean input
def custom_input(prompt):
    readline.set_pre_input_hook(lambda: readline.insert_text(""))
    readline.set_startup_hook(lambda: readline.insert_text(""))
    try:
        return input(prompt)
    finally:
        readline.set_pre_input_hook(None)
        readline.set_startup_hook(None)


def main():

    # create prompt for persona chat
    system_prompts = "".join(
        [
            "너의 이름은 민지.",
            "취미: 영화보기, 요리하기, 쇼핑하기",
            "너는 대화할 때 이모티콘을 자주 써.",
        ]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompts),
            MessagesPlaceholder(variable_name=HISTORY_MEMORY_KEY),
            ("human", "{question}"),
        ]
    )

    # initialize language model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    # initialize memory for chat history
    memory = ConversationBufferMemory(
        memory_key=HISTORY_MEMORY_KEY,
        return_messages=True,
    )
    runnable = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables)
        | itemgetter(HISTORY_MEMORY_KEY)
    )

    # create chain
    chain = runnable | prompt | llm | StrOutputParser()

    while True:
        user_input = custom_input("You: ")

        # no input
        if not user_input:
            continue

        # exit
        if user_input.lower() == "exit":
            print("Assistant: Goodbye!")
            break

        # show chat history
        if user_input.lower() == "show history":
            print("Assistant:", memory.load_memory_variables()[HISTORY_MEMORY_KEY])
            continue

        # invoke chain
        response = chain.invoke({"question": user_input})

        # save chat history
        memory.save_context({"input": user_input}, {"output": response})

        # print response
        print("Assistant:", response)


if __name__ == "__main__":
    main()
