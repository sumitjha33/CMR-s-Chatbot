import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from flask import Flask, render_template, request, jsonify
import re

app = Flask(__name__)

# ✅ Bind host & dynamic port for deployment
PORT = int(os.getenv("PORT", 10000))

# ✅ Configure Google API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY is missing!")
genai.configure(api_key=api_key)

# ✅ Load FAISS vector store (with error handling)
try:
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("cmr_faiss_index", embeddings=embedding_function, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"❌ Error loading FAISS: {e}")
    vectorstore = None

# ✅ Initialize LLM with optimal settings
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)

# ✅ Structured prompt template
prompt_template = PromptTemplate(
    input_variables=["query", "context_text"],
    template="""
You are the **CMR University AI Chatbot**, providing clear, structured, and engaging responses.

### **FORMAT GUIDELINES:**
- **Use Headings (##) & Subheadings (###)**
- **Use Bullet Points (-) & Numbered Lists (1,2,3)**
- **Highlight Key Information in Bold**
- **Separate Sections with (---)**
- **Use Blockquotes (>) for Important Notes**

### **User Query:**
{query}

### **Relevant Information:**
{context_text}

Generate a response using this structured format:
"""
)

llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# ✅ Enhanced formatting function
def enhance_response(text):
    """Formats response with better readability & styling."""

    text = text.replace("**", "<b>").replace("*", "</b>")
    text = text.replace("\n\n", "<br><br>")  # Improve spacing
    text = text.replace("* ", "• ")  # Convert * to bullet points
    text = text.replace("---", "<hr>")  # Divider for sections

    # Apply numbered list formatting
    text = re.sub(r'(\d+)\. ', r'<br><b>\1.</b> ', text)

    # Apply blockquote styling
    text = re.sub(r'>(.*?)\n', r'<blockquote>\1</blockquote>', text)

    return f"""
    <div style='font-family: Arial, sans-serif; font-size: 16px; line-height: 1.6; padding: 10px; color: #333;'>
        {text}
    </div>
    """

# ✅ Chatbot Logic
def cmr_chatbot(query):
    if not vectorstore:
        return enhance_response("❌ No document index found. Please ensure FAISS is correctly set up.")

    # ✅ Optimized FAISS search (reduce `k=5` to `k=3` for speed)
    retrieved_docs = vectorstore.similarity_search(query, k=3)
    if not retrieved_docs:
        return enhance_response("❌ No relevant information found in the CMR document.")

    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    try:
        response = llm_chain.invoke({"query": query, "context_text": context_text})
        return enhance_response(response["text"])
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return enhance_response("❌ AI response failed due to an internal issue.")

# ✅ Flask Routes
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_query = request.json.get("query", "").strip()
        if not user_query:
            return jsonify({"response": "❌ Please enter a valid question."}), 400

        response = cmr_chatbot(user_query)
        return jsonify({"response": response})
    except Exception as e:
        print(f"❌ Server Error: {e}")
        return jsonify({"response": f"❌ Internal Server Error: {str(e)}"}), 500

# ✅ Run Flask with proper host binding
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
