from langchain_openai import OpenAI
# from langchain_google_genai import GoogleGenerativeAI

openai_llm = OpenAI()

# gemini_pro = GoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, max_output_tokens=1024)

response = openai_llm.invoke("Tell me a joke about programming.")

print(response)