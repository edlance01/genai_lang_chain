from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# local_llm = ChatOllama(model="llama2", base_url="http://localhost:11434")
local_llm = ChatOllama(model="deepseek-r1:1.5b", temperature=0)
# faster, non "thinking" model, but less accurate
# ollama pull llama3.2:1b

print("Starting...")
prompt = PromptTemplate.from_template("Explain {concept} in simple terms.")
print("Prompt created...")
local_chain = prompt | local_llm | StrOutputParser()
print("Chain created...")
# use to diagnose slow response
# Stream chunks instead of invoking all at once
for chunk in local_llm.stream("Explain quantum computing simply"):
    print(chunk.content, end="", flush=True)

# Don't use this code, see EL Gemini "Simple Local Ollam Python Example" for more details 
# result = local_chain.invoke({"concept": "quantum computing"})
# print("Result obtained...")
# print(result)
print("Done.")
