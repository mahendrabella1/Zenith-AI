import os
import requests
import streamlit as st
import nltk
import json
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from groq import Groq

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "college-data"  # Store only web page data

if not GROQ_API_KEY:
    raise ValueError("❌ ERROR: Missing Groq API Key. Check your .env file!")

# ✅ Ensure nltk dependency
try:
    nltk.data.find('corpora/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# ✅ Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# ✅ Local storage for college data
COLLEGE_DATA_FILE = "college_data.json"

if not os.path.exists(COLLEGE_DATA_FILE):
    with open(COLLEGE_DATA_FILE, "w") as f:
        json.dump([], f)  # Initialize empty JSON file


# ---------------------------- Helper Functions ----------------------------

def is_valid_url(url):
    """Check if the URL is valid and accessible."""
    try:
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def extract_text_from_webpage(url):
    """Extract text content from a webpage."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    return "\n".join([para.get_text() for para in paragraphs]).strip()


def load_pdf(pdf_path):
    """Load and extract text from a PDF."""
    return PyPDFLoader(pdf_path).load()


def store_college_data(pdf_path, source_name):
    """Store college data locally instead of Pinecone."""
    if "processed_college_data" not in st.session_state:
        st.session_state.processed_college_data = set()

    if source_name in st.session_state.processed_college_data:
        return "✅ College data is already stored locally."

    documents = load_pdf(pdf_path)
    text_data = "\n".join([doc.page_content for doc in documents])

    # ✅ Save to JSON
    with open(COLLEGE_DATA_FILE, "w") as f:
        json.dump({"source": source_name, "content": text_data}, f)

    st.session_state.processed_college_data.add(source_name)
    st.session_state.current_college_source = source_name

    return "✅ College data stored locally."


def store_web_data(url):
    """Process and store embeddings for web pages in Pinecone."""
    if "processed_web_data" not in st.session_state:
        st.session_state.processed_web_data = set()

    if url in st.session_state.processed_web_data:
        return "✅ Web page data is already stored in Pinecone."

    if not is_valid_url(url):
        return "❌ Error: URL is not accessible."

    text_data = extract_text_from_webpage(url)
    if not text_data:
        return "❌ Error: No readable text found."

    # ✅ Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_text(text_data)

    # ✅ Generate embeddings
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectors = [embeddings_model.embed_query(chunk) for chunk in text_chunks]

    # ✅ Store embeddings in Pinecone
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    ids = [f"{url}_{i}" for i in range(len(text_chunks))]
    metadata = [{"text": chunk, "source": url} for chunk in text_chunks]

    index.upsert(vectors=list(zip(ids, vectors, metadata)))

    # ✅ Mark URL as processed
    st.session_state.processed_web_data.add(url)

    return "✅ Web page data stored in Pinecone."


def query_college_data(question):
    """Retrieve and return college data from the local JSON file."""
    with open(COLLEGE_DATA_FILE, "r") as f:
        data = json.load(f)

    if not data or "content" not in data:
        return "❌ No college data found. Please upload a college data PDF."

    return f"📄 College Data:\n\n{data['content']}\n\n🔍 Answer based on this information."


def query_web_data(question):
    """Retrieve relevant information from stored web page embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        docsearch = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings)
    except Exception as e:
        return f"❌ Error: Could not connect to Pinecone index. {str(e)}"

    relevant_docs = docsearch.similarity_search(question, k=5)

    if not relevant_docs:
        return "❌ No relevant web data found."

    retrieved_text = "\n".join([doc.page_content for doc in relevant_docs])

    return f"🌍 Web Data:\n\n{retrieved_text}\n\n🔍 Answer based on this information."


def query_model(question):
    """Use the Groq API model directly."""
    chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": "You are an advanced AI assistant."},
                  {"role": "user", "content": question}],
        model="llama3-70b-8192",
        stream=False,
    )
    return chat_completion.choices[0].message.content


# ---------------------------- Streamlit UI ----------------------------

def main():
    st.set_page_config(page_title="Zenith AI", page_icon="🧠")
    st.title("🧠 Zenith AI - The Ultimate Thinking Machine")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.divider()

        if "current_college_source" not in st.session_state:
            st.session_state.current_college_source = "No file uploaded"

        st.caption(f"Current College Data Source: {st.session_state.current_college_source}")

        option = st.radio("Select knowledge base:", ("Model", "College Data", "Upload PDF", "Enter URL"), index=0)

        if option == "Upload PDF":
            pdf_file = st.file_uploader("Choose PDF file", type=["pdf"])
            if pdf_file:
                temp_path = f"temp_{pdf_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(pdf_file.getbuffer())

                with st.spinner("Processing PDF..."):
                    result = store_college_data(temp_path, pdf_file.name)
                    st.success(result)

        elif option == "Enter URL":
            url = st.text_input("Enter website URL:")
            if st.button("Process URL") and url:
                with st.spinner("Analyzing website content..."):
                    result = store_web_data(url)
                    st.success(result)

    # Main chat interface
    st.subheader("Chat with ZenithAI")

    if prompt := st.chat_input("Ask a question..."):
        st.markdown(f"👤 **You:** {prompt}")

        with st.spinner("🔍 Analyzing..."):
            if option == "Model":
                response = query_model(prompt)
            elif option == "College Data":
                response = query_college_data(prompt)
            else:
                response = query_web_data(prompt)

            st.markdown(f"🤖 **Zenith AI:** {response}")


if __name__ == "__main__":
    main()
