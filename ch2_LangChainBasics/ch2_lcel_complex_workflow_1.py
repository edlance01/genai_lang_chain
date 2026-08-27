from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

'''
In this version we get the analysis but lose the story.  See same file name _2
to see how to get both the story and the analysis in the output.
'''
# Use ChatGoogleGenerativeAI and a current model identifier
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 1st chain: generates a story
story_prompt = PromptTemplate.from_template("Write a story about {topic}.")
story_chain = story_prompt | llm | StrOutputParser()

# 2nd chain: analyzes the story mood
analyze_prompt = PromptTemplate.from_template(
    "Analyze the following story's mood:\n{story}"
)
analyze_chain = analyze_prompt | llm | StrOutputParser()

# Compose LCEL workflow
story_with_analysis = story_chain | analyze_chain

# Execute
response = story_with_analysis.invoke({"topic": "a brave little toaster"})
print(response)
