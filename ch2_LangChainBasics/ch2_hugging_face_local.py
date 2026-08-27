from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


"""
This code uses HuggingFacePipeline, which downloads and runs the TinyLlama
model directly inside your Python environment via PyTorch and the transformers 
library—completely independent of Ollama.
"""
# Create a pipeline with the full Hugging Face repository ID
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=256,
        do_sample=False,
        repetition_penalty=1.03,  # Fixed typo: repetition_penalty
    ),
)

chat_model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Translate the following sentence to French: 'I love programming.'"
    ),
]

# Use .invoke() instead of calling the object directly
ai_msg = chat_model.invoke(messages)
print(ai_msg.content)
