from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# components
prompt = PromptTemplate.from_template("Tell me a joke about {topic}.")
llm = ChatOpenAI(model="gpt-4o")
output_parser = StrOutputParser()

# use LCEL to create chain
chain = prompt | llm | output_parser
# execute
response = chain.invoke({"topic": "programming"})
print(response)
