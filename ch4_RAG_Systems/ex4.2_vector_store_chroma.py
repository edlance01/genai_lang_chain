from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# iniitalize the embeddings model
embeddings_model = OpenAIEmbeddings()
vector_store = Chroma(embedding_function=embeddings_model)

# The vectorstore base class in LangChain provides these operations
docs = [Document(page_content="Content 1"), Document(page_content="Content 2")]
ids = vector_store.add_documents(docs)

# Similarity search . k - number of results to return
results = vector_store.similarity_search("How does LangChain do embeddings", k=1)

# Deletion of documents from the vector store
#vector_store.delete(ids=ids)

# Maintenace marginal relevance search (MRS) is a technique used to improve the quality of search results by considering the relevance of documents in relation to each other. It helps to reduce redundancy and increase diversity in the results returned by a search query. In LangChain, MRS can be implemented by using the `marginal_relevance_search` method provided by the vector store class. This method takes into account the similarity of documents to the query as well as their relevance to each other, allowing for more diverse and informative search results.
results_mrs = vector_store.max_marginal_relevance_search("How does LangChain do embeddings", k=3, fetch_k=10, lambda_mult=0.5) # Controls diversity (0=max diversity, 1=max relevance)


print(f"Results from similarity search: {results}")
print(f"Results from marginal relevance search: {results_mrs}")
