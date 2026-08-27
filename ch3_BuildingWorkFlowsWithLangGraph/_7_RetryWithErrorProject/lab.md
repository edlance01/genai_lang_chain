### `LAB.md`

```markdown
# Hands-On Lab: Modern LangChain Structured Outputs & Self-Correction

## Overview
In this lab, you will learn how to implement robust structured data extraction using modern LangChain components. Rather than relying on legacy output parsers, you will use **Pydantic V2** combined with `.with_structured_output()` and build an automated error-repair workflow using **LangChain Expression Language (LCEL)**.

---

## Architecture Pattern


```

[Raw Input Text] ──> [ChatPromptTemplate] ──> [LLM + Structured Output] ──> [Validated Schema]
│
├── (On Parse Error)
▼
[Error Details + Raw Output] ──> [Repair Chain] ────────┘

```

---

## Exercise 1: Project Setup

### Step 1: Create a Virtual Environment
Open your terminal in VS Code and create a clean isolated Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

```

### Step 2: Set Up Project Files

Create three files in your working directory:

* `.env`
* `requirements.txt`
* `main.py`

### Step 3: Define Dependencies

Add the following package versions to your `requirements.txt` file:

```text
langchain>=0.2.0
langchain-core>=0.2.0
langchain-openai>=0.1.0
pydantic>=2.0.0
python-dotenv>=1.0.0

```

Install them into your virtual environment:

```bash
pip install -r requirements.txt

```

### Step 4: Configure API Key

Add your OpenAI API key to the `.env` file:

```env
OPENAI_API_KEY=your_actual_openai_api_key_here

```

---

## Exercise 2: Implementing the Extraction Script

Paste the following modern LCEL code into `main.py`:

```python
import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. Load environment variables from .env
load_dotenv()


# 2. Define the target schema using Pydantic V2
class ActionItem(BaseModel):
    task: str = Field(description="The action item description")
    assignee: str = Field(description="Person responsible for the task")


class ActionItemList(BaseModel):
    items: List[ActionItem] = Field(description="List of extracted action items")


def run_structured_chain(text: str, structured_llm):
    """Standard LCEL chain execution for structured generation."""
    prompt = ChatPromptTemplate.from_template(
        "Extract all action items from the following text:\n\n{text}"
    )
    chain = prompt | structured_llm
    return chain.invoke({"text": text})


def repair_malformed_output(sample_text: str, bad_completion: str, error_msg: str, structured_llm):
    """LCEL repair chain that uses failure context to self-correct."""
    repair_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert data corrector. Convert the provided raw output into the requested JSON schema."),
        ("human", "Original Input Text:\n{input}\n\nFailed Completion:\n{completion}\n\nValidation Error Details:\n{error}")
    ])
    
    repair_chain = repair_prompt | structured_llm
    return repair_chain.invoke({
        "input": sample_text,
        "completion": bad_completion,
        "error": error_msg
    })


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY missing from environment or .env file.")

    # 3. Instantiate model with native structured output constraint
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(ActionItemList)

    sample_text = "Alice needs to prepare the Q3 report by Friday. Bob will handle the client presentation."

    print("--- Step 1: Running Modern Structured Output Chain ---")
    valid_result = run_structured_chain(sample_text, structured_llm)
    print(f"Parsed Result: {valid_result}\n")

    print("--- Step 2: Simulating Parsing Error & Triggering Self-Correction ---")
    bad_raw_completion = (
        "Here are the items:\n- Task: Q3 Report, Assignee: Alice\n- Task: Presentation, Assignee: Bob"
    )

    try:
        # Intentionally force a validation check to capture failure details
        ActionItemList.model_validate_json(bad_raw_completion)
    except ValidationError as err:
        print("Caught parsing error successfully!")
        
        repaired_result = repair_malformed_output(
            sample_text=sample_text,
            bad_completion=bad_raw_completion,
            error_msg=str(err),
            structured_llm=structured_llm
        )
        print("\nSuccessfully Repaired Output:")
        print(repaired_result)


if __name__ == "__main__":
    main()

```

---

## Exercise 3: Execution and Verification

### Step 1: Run the Lab Script

Execute your completed python script:

```bash
python main.py

```

### Step 2: Verify Results

Confirm your output matches the expected behavior:

1. **Step 1 Result**: An `ActionItemList` instance containing clean `ActionItem` Pydantic objects.
2. **Step 2 Result**: A caught `ValidationError`, followed by the repair chain output converting `bad_raw_completion` into a valid `ActionItemList`.

---

## Lab Challenge

Modify `ActionItem` in `main.py` to include a `due_date: str` field. Update the sample prompt text with due dates and ensure the self-correction logic correctly repairs outputs with missing or malformed dates.

```

```