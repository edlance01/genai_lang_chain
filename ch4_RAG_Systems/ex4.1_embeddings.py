from langchain_openai import OpenAIEmbeddings

# Initialize the embeddings model
embeddings_model = OpenAIEmbeddings()

# Create embeddings for the original example sentences
text1 = "The cat sat on the mat"
text2 = "A feline rested on the carpet"
text3 = "Python is a programming language"

# Get embeddings using LangChain
embeddings = embeddings_model.embed_documents([text1, text2, text3])

# These similar sentences will have similar embeddings
embedding1 = embeddings[0] # Embedding for "The cat sat on the mat"
embedding2 = embeddings[1] # Embedding for "A feline rested on the carpet"
embedding3 = embeddings[2] # Embedding for "Python is a programming language"

# Output shows 3 documents with their embedding dimensions
print(f"Number of documents: {len(embeddings)}")
print(f"Dimensions per embedding: {len(embeddings[0])}")
# Typically 1536 dimensions with OpenAI's embeddings