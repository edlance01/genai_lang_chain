from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

class JobApplicationState(TypedDict):
    job_description: str
    is_suitable: bool
    application: str

def analyze_job_description(state):
    print("...analyinzing a provided job description...")
    print(state)
    return {"is_suitable": len(state["job_description"]) > 100}

def generate_application(state):
    print("...generating application...")
    print(state)
    return {"application": "some_fake_application"}


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

res = graph.invoke({"job_description": "fake_jd"})
print(res)
