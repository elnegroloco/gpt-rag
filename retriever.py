import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError(" OPENAI_API_KEY is missing. Please check your .env file.")

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata4")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key= openai_api_key)

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define the user's question
query = "List risks in building construction?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3 ,"fetch_k": 20, "lambda_mult": 0.5},
)
relevant_docs = retriever.invoke(query)


# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
