from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

class IsSuitableJobEnum(str, Enum):
    YES = "YES"
    NO = "NO"

class DecisionSchema(BaseModel):
    decision: IsSuitableJobEnum = Field(description="Selected decision, either YES or NO")

parser = PydanticOutputParser(pydantic_object=DecisionSchema)

prompt = PromptTemplate(
    template="Given a job description, decide whether it suits a junior Java developer.\n\nJOB DESCRIPTION:\n{job_description}\n\n{format_instructions}",
    input_variables=["job_description"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# uses GEMINI API KEY from environment
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

job_description ="We are looking for an entry-level Junior Java Developer to assist in building, testing, and maintaining core backend applications. You will collaborate with senior developers to write clean, efficient code using Java and modern frameworks like Spring Boot. The ideal candidate has a strong foundation in Object-Oriented Programming and an eagerness to learn scalable software design."


# chain the llm with the output parser
chain = prompt | llm | parser

# return just a string (not AI Message Object)
response = chain.invoke({"job_description": job_description})

print(response.decision)
print(response.decision.value)
