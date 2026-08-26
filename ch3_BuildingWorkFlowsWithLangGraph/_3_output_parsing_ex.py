from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI


class JobApplicationState(TypedDict):
    job_description: str
    is_suitable: bool
    application: str
    actions: List[str]


def analyze_job_description(state):
    print("...analyzing a provided job description...")
    print(state)
    return {"is_suitable": len(state["job_description"]) > 100}


def generate_application(state: JobApplicationState, config: RunnableConfig):
    print("...generating application...")
    model_provider = config["configurable"].get("model_provider", "Google")
    model_name = config["configurable"].get("model_name", "gemini-2.5-flash")

    print(f"... generating application with {model_provider} and {model_name} ...")

    # Initialize the standard Google GenAI model
    llm = ChatGoogleGenerativeAI(model=model_name)

    # Example invoke call:
    # response = llm.invoke(f"Write a brief application snippet for: {state['job_description']}")

    print(state)
    return {"application": "some_fake_application", "actions": ["action2", "action3"]}


builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_application)
builder.add_edge(START, "analyze_job_description")
builder.add_edge("analyze_job_description", "generate_application")
builder.add_edge("generate_application", END)

graph = builder.compile()

graph_image = graph.get_graph().draw_mermaid_png()

with open("images/graph_state_management.png", "wb") as f:
    f.write(graph_image)

print("\n--- Second Result config object added ---")
result2 = graph.invoke(
    {"job_description": "Math Instructor"},
    config={
        "configurable": {"model_provider": "Google", "model_name": "gemini-2.5-flash"}
    },
)
print(result2)
