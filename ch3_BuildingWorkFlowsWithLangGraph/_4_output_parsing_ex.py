from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# uses GEMINI API KEY from environment
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

job_description: str = (
    "We are looking for an entry-level Junior Java Developer to assist in building, testing, and maintaining core backend applications. You will collaborate with senior developers to write clean, efficient code using Java and modern frameworks like Spring Boot. The ideal candidate has a strong foundation in Object-Oriented Programming and an eagerness to learn scalable software design."
)


prompt_template_enum = (
    "Given a job description, decide whether it suits a junior Java developer.\n"
    f"\nJOB DESCRIPTION:\n{job_description}\n\nAnswer only YES or NO."
)

# chain the llm with the output parser
chain = llm | StrOutputParser()

# return just a string (not AI Message Object)
response = chain.invoke(prompt_template_enum).strip().upper()

print(response)
