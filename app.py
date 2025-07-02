import os 
import streamlit as st
st.set_page_config(page_title="GRC Chatbot", page_icon="🛡️", layout="centered")
st.markdown("""
    <style>
    .main {
        background-color: #0F1117;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTextInput>div>div>input {
        background-color: #1c1e26;
        color: white;
    }
    .stTextInput label, .stTextArea label {
        color: #ffffff;
    }
    .chat-box {
        background-color: #1c1e26;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        border: 1px solid #303030;
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
from transformers import pipeline
# Load summarizer pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load GRC documents from cyber_docs folder
docs = []
for file in ["cyber_docs/nist.txt", "cyber_docs/iso27001.txt", "cyber_docs/gdpr.txt"]:
    loader = TextLoader(file)
    docs.extend(loader.load())

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

# Use HuggingFace Local Embeddings (NO OpenAI key needed)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store using FAISS
vectorstore = FAISS.from_documents(split_docs, embeddings)

# Simple LLM-like function (Fetch top similar docs)
def simple_llm(query):
    # Search top matching chunks (across all documents)
    matched_docs = vectorstore.similarity_search(query, k=5)

    # Try to filter results: Prefer documents matching query keywords
    keyword = ""
    if "gdpr" in query.lower():
        keyword = "gdpr"
    elif "nist" in query.lower():
        keyword = "nist"
    elif "iso" in query.lower() or "27001" in query:
        keyword = "iso"

    filtered_docs = [
        doc for doc in matched_docs if keyword in doc.metadata['source'].lower()
    ]

    # Fallback: If filter returns nothing, use original results
    if not filtered_docs:
        filtered_docs = matched_docs

    combined_text = " ".join([doc.page_content for doc in filtered_docs])
    summary = summarizer(combined_text, max_length=200, min_length=50, do_sample=False)
    return summary[0]['summary_text']


# Streamlit UI
st.title("🛡️ AI Chatbot for GRC")
uploaded_file = st.file_uploader("📁 Upload a new GRC .txt file", type="txt")
if uploaded_file is not None:
    file_path = os.path.join("cyber_docs", uploaded_file.name)

    # Save uploaded file to cyber_docs folder
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ {uploaded_file.name} uploaded successfully!")
        # Load the uploaded file
    loader = TextLoader(file_path)
    new_docs = loader.load()
    docs.extend(new_docs)

    # Re-split all documents (old + new)
    split_docs = text_splitter.split_documents(docs)

    # Rebuild the vectorstore with updated docs
    vectorstore = FAISS.from_documents(split_docs, embeddings)

query = st.text_input("Ask your GRC question:")

if st.button("🧹 Clear Chat"):
    st.session_state.history = []
    st.experimental_rerun()

st.markdown("""
**💡 Example Queries:**  
- What are the functions of NIST?  
- Explain GDPR principles.  
- What is ISO 27001 risk assessment?
""")

if query:
    try:
        result = simple_llm(query)

        st.session_state.history.append((query,result))
        st.subheader("🔍 Answer:")
        st.write(result)

        if st.session_state.history:
            st.markdown("---")
            st.subheader("💬 Chat History")

            for i, (q, a) in enumerate(reversed(st.session_state.history), 1):
                st.markdown(f"""
    <div class='chat-box'>
        <p><b>🧑 You:</b> {q}</p>
        <p><b>🤖 Bot:</b> {a}</p>
    </div>
""", unsafe_allow_html=True)

    except Exception as e:
       st.error(f"An error occurred: {e}")
else:
    st.info("Please enter a query related to GRC (NIST, ISO 27001, GDPR).")
