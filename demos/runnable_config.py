from typing import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# 1. State Definition
class State(TypedDict):
    greeting: str
    count: int


# 2. Node Functions accepting RunnableConfig
def greeting_node(state: State, config: RunnableConfig) -> dict:
    user_name = config.get("configurable", {}).get("user_name", "Guest")
    return {"greeting": f"Hello, {user_name}!"}


def counter_node(state: State, config: RunnableConfig) -> dict:
    tags = config.get("tags", [])
    current_count = state.get("count", 0) + 1
    print(f" [Node Execution] Tags: {tags} | Updated Count: {current_count}")
    return {"count": current_count}


# 3. Graph Assembly
builder = StateGraph(State)
builder.add_node("greeter", greeting_node)
builder.add_node("counter", counter_node)

builder.add_edge(START, "greeter")
builder.add_edge("greeter", "counter")
builder.add_edge("counter", END)

graph = builder.compile(checkpointer=MemorySaver())

# 4. Execution Examples
if __name__ == "__main__":

    no_name_config: RunnableConfig = {
        "configurable": {"thread_id": "thread-101"},
        "tags": ["prod-test"],
    }

    # Config for Alice
    alice_config: RunnableConfig = {
        "configurable": {"thread_id": "thread-101", "user_name": "Alice"},
        "tags": ["prod-test"],
    }

    print("=== Run 1 (Alice) ===")
    res1 = graph.invoke({"count": 0}, config=no_name_config)
    print("Output:", res1)

    print("\n=== Run 2 (Alice - Increments existing thread) ===")
    res2 = graph.invoke({}, config=alice_config)
    print("Output:", res2)

    # Config for Bob
    bob_config: RunnableConfig = {
        "configurable": {"thread_id": "thread-102", "user_name": "Bob"},
        "tags": ["prod-test"],
    }

    print("\n=== Run 3 (Bob - Separate state thread) ===")
    res3 = graph.invoke({"count": 0}, config=bob_config)
    print("Output:", res3)
