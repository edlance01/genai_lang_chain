**Why this section feels confusing and light**

This section skips over the core reason *why* LangGraph needs reducers in the first place, jumping straight to syntax.

In standard Python code, when you modify a variable, you overwrite it. In LangGraph, workflows are built out of nodes that run during "supersteps"—and multiple nodes can even run in parallel. A **reducer** is simply a rule that tells LangGraph: *"When a node returns data for a state key, how should that new value (`right`) be combined with the existing value (`left`)?"*

---

**Explaining the 3 Options from Your Text**

**1. Default behavior (Overwrite)**

* **Syntax:** `actions: list[str]`
* **How it works:** Every time a node returns `{"actions": ["send_email"]}`, it wipes out whatever was previously in `actions` and replaces it.
* **Problem:** You lose history unless the node manually fetches the old list, appends to it, and returns the whole thing.

**2. Built-in `operator.add` (Concatenate)**

* **Syntax:** `actions: Annotated[list[str], add]`
* **How it works:** When a node returns `{"actions": ["send_email"]}`, LangGraph uses Python's `+` operator behind the scenes (`left + right`). It appends the new list to the existing list.
* **Requirement:** Nodes **must** return a list (e.g., `["send_email"]`). Passing a string directly will cause a type error or scatter individual characters into the list.

**3. Custom Reducer (`my_reducer`)**

* **Syntax:** `actions: Annotated[list[str], my_reducer]`
* **How it works:** It adds defensive logic to make the state updates more forgiving.
* **Why use it:** It lets nodes return either a single string (`"send_email"`) or a list of strings (`["send_email"]`), converting the single string into a list before appending it to `left`.

---

**Most Common Reducer Practices in LangGraph**

In real-world applications, you will encounter two primary reducer patterns:

* **`add_messages` (By far the most common):**
Since LLM apps are conversational, state almost always needs a message history. `add_messages` does much more than standard list concatenation (`+`):
* Appends new messages to the end of the history.
* Overwrites an existing message in-place if a new message has the same `id` (crucial for streaming or updating assistant responses).
* Deletes messages from state if a `RemoveMessage` object is returned.


* **Inheriting from `MessagesState`:**
Instead of writing `messages: Annotated[list[AnyMessage], add_messages]` manually, the industry standard for message-based graphs is simply extending `MessagesState`:

```python
from langgraph.graph import MessagesState

# Automatically includes 'messages' key with 'add_messages' reducer attached
class JobApplicationState(MessagesState):
    user_id: str
    status: str

```

* **Standard `add` for lists:**
For simple accumulators (like tracking tools used, document IDs retrieved, or sub-task results), developers rely on `operator.add`.