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

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

try:
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("cmr_faiss_index", embeddings=embedding_function, allow_dangerous_deserialization=True)
except Exception:
    vectorstore = None

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)

prompt_template = PromptTemplate(
    input_variables=["query", "context_text"],
    template="""
You are the **CMR University AI Chatbot**, designed to provide clear and structured answers.

### **OUTPUT FORMAT INSTRUCTIONS (FOLLOW EXACTLY):**
1. Use a consistent hierarchy with main headings (##) and subheadings (###)
2. For sequential processes or steps, use a SINGLE ordered list with items numbered 1, 2, 3, etc.
3. Under each ordered list item, place related bullet points in a SINGLE unordered list
4. Format all bullet points with a single dash (-) at the start
5. Bold important phrases with **double asterisks**
6. Format all links as [text](url)
7. Use --- for horizontal dividers between major sections
8. Ensure consistent spacing: one blank line before headings, lists, and after paragraphs

### **User Query:**
{query}

### **Document Information:**
{context_text}

Follow the format instructions precisely. Generate a structured response:
""",
)

llm_chain = LLMChain(llm=llm, prompt=prompt_template)

def convert_to_html(markdown_text):
    html_parts = ['<div class="chatbot-response">']
    sections = re.split(r'^---+$', markdown_text, flags=re.MULTILINE)
    
    for idx, section in enumerate(sections):
        if not section.strip():
            continue
        
        section_html = []
        lines = section.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('## '):
                section_html.append(f'<h2 class="response-heading">{line[3:].strip()}</h2>')
            elif line.startswith('### '):
                section_html.append(f'<h3 class="response-subheading">{line[4:].strip()}</h3>')
            elif re.match(r'^\d+\.\s+', line):
                ol_html = ['<ol class="ordered-list">']
                while i < len(lines) and re.match(r'^\d+\.\s+', lines[i].strip()):
                    item_text = re.sub(r'^\d+\.\s+', '', lines[i].strip())
                    ol_html.append(f'<li class="list-item">{item_text}</li>')
                    i += 1
                ol_html.append('</ol>')
                section_html.append('\n'.join(ol_html))
                i -= 1
            elif line.startswith('- '):
                ul_html = ['<ul class="bullet-list">']
                while i < len(lines) and lines[i].strip().startswith('- '):
                    bullet = lines[i].strip()[2:]
                    ul_html.append(f'<li class="list-item">{bullet}</li>')
                    i += 1
                ul_html.append('</ul>')
                section_html.append('\n'.join(ul_html))
                i -= 1
            elif line:
                section_html.append(f'<p class="response-paragraph">{line}</p>')
            
            i += 1
        
        if idx > 0:
            html_parts.append('<hr class="section-divider">')
        html_parts.append('\n'.join(section_html))
    
    html_parts.append('</div>')
    return '\n'.join(html_parts)

def cmr_chatbot(query):
    if not vectorstore:
        return "❌ No document index found. Please ensure FAISS is correctly set up."

    retrieved_docs = vectorstore.similarity_search(query, k=5)
    if not retrieved_docs:
        return "❌ No relevant information found in the CMR document."

    context_text = "\n\n".join([f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(retrieved_docs)])
    raw_response = llm_chain.run({"query": query, "context_text": context_text})
    return convert_to_html(raw_response)

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
        return jsonify({"response": f"❌ Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)