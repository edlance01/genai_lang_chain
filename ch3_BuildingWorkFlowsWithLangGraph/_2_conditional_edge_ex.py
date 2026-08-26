from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END

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
def is_suitable_condition(state: JobApplicationState) -> Literal["generate_application", "__end__"]:
    if state.get("is_suitable"):
        return "generate_application"
    return END

builder.add_edge(START, "analyze_job_description")
builder.add_conditional_edges("analyze_job_description", is_suitable_condition)
builder.add_edge("generate_application", END)
graph = builder.compile()

graph_image = graph.get_graph().draw_mermaid_png()

with open("images/graph_conditional_edge.png", "wb") as f:
    f.write(graph_image)

res = graph.invoke({"job_description": "fake_jd"})
print(res)

valid_job_description = "We are seeking a driven Software Engineer to build scalable APIs and optimize backend performance. Apply today to join our team!"

valid_job_result = graph.invoke({"job_description": valid_job_description})
print(valid_job_result)

