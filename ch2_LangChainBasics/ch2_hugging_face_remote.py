from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Loads HF_TOKEN into os.environ
load_dotenv()

# The endpoint automatically detects HF_TOKEN
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation",
    max_new_tokens=256,
)

chat_model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(
        content="You are an AI expert."
    ),
    HumanMessage(
        content="Explain inference in machine learning, give an example and a metaphor."
    ),
]

ai_msg = chat_model.invoke(messages)
print(ai_msg.content)
