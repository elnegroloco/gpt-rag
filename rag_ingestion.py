import os
import json
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing. Please check your .env file.")

# Define directories
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books3")  # Folder containing books
db_dir = os.path.join(current_dir, "db")  # Database folder
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata5")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

# Initialize LLM for metadata tagging
llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)

# Function to generate metadata using GPT
def generate_metadata(text, filename):
    prompt = (
        "You are an expert project analyst."
        "\n\nRead the following construction-related document excerpt and extract metadata as a JSON object with these fields:"
        "\n- project_type (bridge, school, road, tunnel, etc.)"
        "\n- sector (transport, education, housing, etc.)"
        "\n- country_or_region"
        "\n- document_type (risk_register, strategy, progress_report, etc.)"
        "\n- project_name (if stated)"
        "\n\nIf a field is not available, say 'unknown'. Output only valid JSON."
        f"\n\nText:\n{text[:1500]}"
    )
    try:
        response = llm.invoke(prompt)
        print(f"\n🧠 GPT Raw Output for {filename} ->\n{response.content.strip()}\n")
        return json.loads(response.content.strip())
    except Exception as e:
        print(f"❌ Failed to extract metadata for {filename}: {e}")
        return {}

# Check if Chroma vector store exists
if not os.path.exists(persistent_directory):
    print("\n--- Vector store not found. Initializing Chroma DB ---")

    if not os.path.exists(books_dir):
        raise FileNotFoundError(f" The directory {books_dir} does not exist.")

    book_files = [f for f in os.listdir(books_dir) if f.endswith((".txt", ".pdf"))]

    if not book_files:
        raise ValueError(" No valid .txt or .pdf files found in the books directory.")

    print(f"📚 Found {len(book_files)} files: {book_files}")

    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)

        if book_file.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8", errors="ignore")
        elif book_file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            continue

        try:
            book_docs = loader.load()
            preview_text = book_docs[0].page_content[:1500]
            metadata = generate_metadata(preview_text, book_file)
            metadata.update({"source": book_file})

            # Fallback: Guess project_type from filename
            if metadata.get("project_type", "unknown") == "unknown":
                if "bridge" in book_file.lower():
                    metadata["project_type"] = "bridge"
                elif "school" in book_file.lower():
                    metadata["project_type"] = "school"
                elif "rail" in book_file.lower():
                    metadata["project_type"] = "rail"

            print(f"📄 Final metadata for {book_file}:\n{json.dumps(metadata, indent=2)}\n")

            for doc in book_docs:
                doc.metadata = metadata
                documents.append(doc)
        except Exception as e:
            print(f"❌ Error reading {book_file}: {e}")

    if not documents:
        raise ValueError("❌ No documents were successfully loaded.")

    print("\n--- Using Recursive Character-based Splitting ---")
    rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = rec_char_splitter.split_documents(documents)

    print("\n--- Document Chunks Information ---")
    print(f"✅ Number of document chunks: {len(docs)}")
    print(f"✅ Sample chunk:\n{docs[0].page_content}\n")
    print(f"✅ Metadata for sample chunk:\n{json.dumps(docs[0].metadata, indent=2)}\n")

    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    print("Finished creating embeddings!")

    print("\n--- Creating and saving vector store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("Vector store created and persisted successfully!")

else:
    print("✅ Vector store already exists. No need to initialize.")
