from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Python is a popular programming language used for data science and machine learning.",
    "Java is widely used for building enterprise-scale applications.",
    "JavaScript is mainly used for web development and frontend applications.",
    "C++ is a high-performance language used in system programming and game development.",
    "SQL is used to manage and query relational databases."
]

query = "tell me about python programming"

# Generate embeddings
doc_embeddings = np.array(embedding.embed_documents(documents))   # (5, dim)
query_embedding = np.array(embedding.embed_query(query)).reshape(1, -1)  # (1, dim)

# Cosine similarity
scores = cosine_similarity(query_embedding, doc_embeddings)[0]

# Best match
index, score = max(enumerate(scores), key=lambda x: x[1])

print("Query:", query)
print("Most relevant document:", documents[index])
print("Similarity score:", score)
