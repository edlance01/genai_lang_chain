import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

# Load environment configuration
load_dotenv()


# 1. Define target schema using Pydantic V2
class ActionItem(BaseModel):
    task: str = Field(description="The action item description")
    assignee: str = Field(description="Person responsible for the task")


class ActionItemList(BaseModel):
    items: List[ActionItem] = Field(description="List of extracted action items")


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY missing from environment or .env file.")

    # 2. Modern LLM initialization with native structured output
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(ActionItemList)

    # 3. Prompt setup
    prompt = ChatPromptTemplate.from_template(
        "Extract all action items from the following text:\n\n{text}"
    )

    # Combine prompt with modern structured LLM (replaces PydanticOutputParser)
    chain = prompt | structured_llm

    sample_text = "Alice needs to prepare the Q3 report by Friday. Bob will handle the client presentation."

    print("--- 1. Modern Structured Output Execution ---")
    result = chain.invoke({"text": sample_text})
    print(result)

    # 4. Modern Repair Pattern (Handling errors when parsing arbitrary string outputs)
    print("\n--- 2. Repairing Bad Input using LCEL ---")

    bad_raw_completion = "Here are the items:\n- Task: Q3 Report, Assignee: Alice\n- Task: Presentation, Assignee: Bob"

    # Try standard manual schema parsing to trigger validation failure
    try:
        ActionItemList.model_validate_json(bad_raw_completion)
    except ValidationError as error:
        print("Caught parsing error successfully!")

        # Fix prompt combining original bad output + exception details
        repair_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert data corrector. Convert the provided completion into the exact JSON schema requested.",
                ),
                (
                    "human",
                    "Original Input: {input}\nBad Completion: {completion}\nError: {error}",
                ),
            ]
        )

        repair_chain = repair_prompt | structured_llm

        # Invoke repair chain directly
        repaired_result = repair_chain.invoke(
            {
                "input": sample_text,
                "completion": bad_raw_completion,
                "error": str(error),
            }
        )

        print("\nSuccessfully Repaired Result:")
        print(repaired_result)


if __name__ == "__main__":
    main()
