import os
from typing import Annotated, List, Optional
import operator
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Ensure API Key is available
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"


# 1. Pydantic Schemas
class UserAnalysis(BaseModel):
    user_name: str = Field(description="Name of the user")
    sentiment: str = Field(
        description="Overall sentiment: Positive, Neutral, or Negative"
    )
    key_intentions: List[str] = Field(
        description="List of core user intentions extracted from text"
    )
    urgency_score: int = Field(
        description="Urgency scale from 1 (low) to 5 (critical)", ge=1, le=5
    )


class AgentState(BaseModel):
    raw_input: str
    analysis: Optional[UserAnalysis] = None
    execution_logs: Annotated[List[str], operator.add] = Field(default_factory=list)


# 2. Node Functions
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def extract_insights_node(state: AgentState) -> dict:
    structured_llm = llm.with_structured_output(UserAnalysis)
    result: UserAnalysis = structured_llm.invoke(state.raw_input)
    return {
        "analysis": result,
        "execution_logs": [f"Extracted info for user: {result.user_name}"],
    }


def route_action_node(state: AgentState) -> dict:
    analysis = state.analysis

    if analysis.urgency_score >= 4:
        action = f"URGENT: Escalated support ticket for {analysis.user_name}."
    else:
        action = f"ROUTINE: Logged feedback for {analysis.user_name}."

    return {"execution_logs": [action]}


# 3. Build Graph
builder = StateGraph(AgentState)
builder.add_node("extractor", extract_insights_node)
builder.add_node("action_router", route_action_node)

builder.add_edge(START, "extractor")
builder.add_edge("extractor", "action_router")
builder.add_edge("action_router", END)

graph = builder.compile()

# 4. Run Execution
if __name__ == "__main__":
    sample_ticket = (
        "Hi, I am Sarah Jenkins. Your system crashed and deleted my work. "
        "I need someone to recover my data immediately! This is blocking my release."
    )

    initial_state = {"raw_input": sample_ticket}
    final_output = graph.invoke(initial_state)

    print("\n--- EXTRACTED PYDANTIC OBJECT ---")
    print(f"User: {final_output['analysis'].user_name}")
    print(f"Sentiment: {final_output['analysis'].sentiment}")
    print(f"Urgency: {final_output['analysis'].urgency_score}/5")
    print(f"Intentions: {final_output['analysis'].key_intentions}")

    print("\n--- EXECUTION LOGS ---")
    for log in final_output["execution_logs"]:
        print(f"- {log}")
