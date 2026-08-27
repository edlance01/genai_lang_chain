from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

# Should fail ever second time
class MessagesIterator:
    def __init__(self):
        self._count = 0

    def __iter__(self):
        return self

    def __next__(self):
        self._count += 1
        if self._count % 2 == 1:
            raise ValueError("Something went wrong")
        return AIMessage(content="False")

fake_llm = GenericFakeChatModel(messages=MessagesIterator())
