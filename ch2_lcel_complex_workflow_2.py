from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough


# Use ChatGoogleGenerativeAI and a current model identifier
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

story_prompt = PromptTemplate.from_template("Write a story about {topic}.")
story_chain = story_prompt | llm | StrOutputParser()

analyze_prompt = PromptTemplate.from_template(
    "Analyze the following story's mood:\n{story}"
)
analyze_chain = analyze_prompt | llm | StrOutputParser()

enhanced_chain = RunnablePassthrough.assign(
    story = story_chain
).assign(
    analysis = analyze_chain
)

# execute
response = enhanced_chain.invoke({"topic": "a brave little toaster"})
                                 
print(response.keys())

