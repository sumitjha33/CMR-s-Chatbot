import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from flask import Flask, render_template, request, jsonify
import logging
import re

# Enable logging for debugging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

try:
    # Load FAISS index
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("cmr_faiss_index", embeddings=embedding_function, allow_dangerous_deserialization=True)
    logging.info("✅ FAISS index loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading FAISS index: {str(e)}")

# Load Gemini LLM using LangChain
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)  # Very low temperature for consistency

# Improved prompt template with precise formatting instructions
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

### **EXAMPLE FORMAT:**
```
## Main Heading

### Subheading

1. **First Step Title:**
   - Detail point one
   - Detail point two

2. **Second Step Title:**
   - Another detail
   - More information
   
---

### Another Section
- Bullet point
- Another bullet point
```

### **User Query:**
{query}

### **Document Information:**
{context_text}

Follow the format instructions precisely. Generate a structured response:
""",
)

# Create LangChain LLM Chain
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

def convert_to_html(markdown_text):
    """
    Convert markdown-formatted text to well-structured HTML with enhanced styling.
    """
    html_parts = []
    # Wrap the entire response in a container
    html_parts.append('<div class="chatbot-response">')
    
    # Split content by major sections (using horizontal rule markers)
    sections = re.split(r'^---+$', markdown_text, flags=re.MULTILINE)
    
    for idx, section in enumerate(sections):
        if not section.strip():
            continue
            
        section_html = []
        lines = section.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Headings with classes for styling
            if line.startswith('## '):
                section_html.append(f'<h2 class="response-heading">{line[3:].strip()}</h2>')
            elif line.startswith('### '):
                section_html.append(f'<h3 class="response-subheading">{line[4:].strip()}</h3>')
                
            # Process ordered lists
            elif re.match(r'^\d+\.\s+', line):
                ol_items = []
                current_idx = i
                # Collect all ordered list items
                while current_idx < len(lines) and re.match(r'^\d+\.\s+', lines[current_idx].strip()):
                    ol_items.append(current_idx)
                    next_idx = current_idx + 1
                    # Skip any bullet points under this ordered item
                    while next_idx < len(lines) and lines[next_idx].strip().startswith('- '):
                        next_idx += 1
                    current_idx = next_idx
                
                ol_html = ['<ol class="ordered-list">']
                for item_idx in ol_items:
                    item_line = lines[item_idx].strip()
                    # Remove the ordered number and format bold text
                    item_text = re.sub(r'^\d+\.\s+', '', item_line)
                    item_text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', item_text)
                    
                    ol_html.append(f'<li class="list-item">{item_text}')
                    
                    # Process any bullet points nested under this ordered item
                    bullet_idx = item_idx + 1
                    bullets = []
                    while bullet_idx < len(lines) and lines[bullet_idx].strip().startswith('- '):
                        bullet = lines[bullet_idx].strip()[2:]
                        bullet = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', bullet)
                        bullet = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', bullet)
                        bullets.append(bullet)
                        bullet_idx += 1
                    
                    if bullets:
                        ul_html = ['<ul class="bullet-list">']
                        for bullet in bullets:
                            ul_html.append(f'<li class="list-item">{bullet}</li>')
                        ul_html.append('</ul>')
                        ol_html.append('\n'.join(ul_html))
                    
                    ol_html.append('</li>')
                ol_html.append('</ol>')
                section_html.append('\n'.join(ol_html))
                i = current_idx - 1
            
            # Process unordered bullet lists
            elif line.startswith('- '):
                ul_html = ['<ul class="bullet-list">']
                bullet = line[2:]
                bullet = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', bullet)
                bullet = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', bullet)
                ul_html.append(f'<li class="list-item">{bullet}</li>')
                
                next_idx = i + 1
                while next_idx < len(lines) and lines[next_idx].strip().startswith('- '):
                    next_bullet = lines[next_idx].strip()[2:]
                    next_bullet = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', next_bullet)
                    next_bullet = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', next_bullet)
                    ul_html.append(f'<li class="list-item">{next_bullet}</li>')
                    next_idx += 1
                ul_html.append('</ul>')
                section_html.append('\n'.join(ul_html))
                i = next_idx - 1
                
            # Process paragraphs
            elif line:
                line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
                line = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', line)
                section_html.append(f'<p class="response-paragraph">{line}</p>')
            
            i += 1
        
        # Add a horizontal divider between sections (except before the first section)
        if idx > 0:
            html_parts.append('<hr class="section-divider">')
        html_parts.append('\n'.join(section_html))
    
    html_parts.append('</div>')
    return '\n'.join(html_parts)

def cmr_chatbot(query):
    """Search FAISS for CMR-related context and generate a structured response"""
    if not vectorstore:
        return "❌ No document index found. Please ensure FAISS is correctly set up."

    # Search FAISS index
    retrieved_docs = vectorstore.similarity_search(query, k=10)
    if not retrieved_docs:
        return "❌ No relevant information found in the CMR document."

    # Extract text from retrieved documents
    context_text = "\n\n".join([f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(retrieved_docs)])
    logging.debug("\n=== DEBUG: Retrieved FAISS Content ===")
    logging.debug(context_text)
    logging.debug("======================================\n")

    # Generate raw response with LLM
    raw_response = llm_chain.run({"query": query, "context_text": context_text})
    logging.debug("\n=== DEBUG: Raw LLM Response ===")
    logging.debug(raw_response)
    logging.debug("======================================\n")

    # Convert the markdown response to properly formatted HTML
    html_response = convert_to_html(raw_response)
    logging.debug("\n=== DEBUG: HTML Response ===")
    logging.debug(html_response)
    logging.debug("======================================\n")
    
    return html_response

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_query = request.json.get("query", "").strip()
        if not user_query:
            return jsonify({"response": "❌ Please enter a valid question."}), 400

        # Generate HTML response
        response = cmr_chatbot(user_query)
        return jsonify({"response": response})

    except Exception as e:
        logging.error(f"❌ Error in /chat endpoint: {str(e)}")
        return jsonify({"response": f"❌ Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)