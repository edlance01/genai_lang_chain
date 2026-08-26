from typing_extensions import TypedDict, List
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langchain_core.runnables import RunnableConfig

class JobApplicationState(TypedDict):
    job_description: str
    is_suitable: bool
    application: str
    actions: List[str]

def analyze_job_description(state):
    print("...analyinzing a provided job description...")
    print(state)
    return {"is_suitable": len(state["job_description"]) > 100}

def generate_application(state: JobApplicationState, config: RunnableConfig):
    print("...generating application...")
    model_provider = config["configurable"].get("model_provider", "Google")
    model_name = config["configurable"].get("model_name", "gemini-2.0-flash-lite")
    print(f"... generating application with {model_provider} and {model_name} ...")
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

# print("---First Result no config object added---")
# res = graph.invoke({"job_description": "fake_jd"})
# print(res)

print("\n--- Second Result config object added ---")
result2 = graph.invoke({"job_description": "Math Instructor"}, 
                       config={"configurable": {"model_provider": "OpenAI", "model_name": "gpt-4o"}})
print(result2)
