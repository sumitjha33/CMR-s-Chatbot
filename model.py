import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

# Set Google Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAvXNg4_SsgUTLL30mQzLbM0cbP48RvoFc"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the PDF (Ignoring Images)
pdf_path = "D:/fitness/main/CMR.pdf"  # Updated to use the uploaded PDF file
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Extract only text from the PDF (Ignoring Images)
text_content = []
for doc in documents:
    if doc.page_content.strip():  # Ignore empty content
        text_content.append(doc.page_content)

# Split text into smaller chunks for better embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text("\n".join(text_content))  # Joining extracted text before splitting

# Load local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings (returns a numpy array)
embedded_chunks = model.encode(chunks, batch_size=200, show_progress_bar=True)

# Convert embeddings to float32 (FAISS requires it)
embedded_chunks = np.array(embedded_chunks, dtype=np.float32)

# Initialize a LangChain-compatible embedding function
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS index
vectorstore = FAISS.from_texts(chunks, embedding_function)

print("FAISS index created successfully!")

# Save FAISS index for later use
vectorstore.save_local("cmr_faiss_index")
