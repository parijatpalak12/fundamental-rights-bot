# 📌 Fundamental Rights Chatbot (NyayaBot)

An AI-powered chatbot that answers questions related to **Part III of the Indian Constitution (Fundamental Rights)** using **Retrieval-Augmented Generation (RAG)**. This project helps users understand constitutional provisions through accurate, context-based responses.

---

## 🌟 Features

- Intelligent Question Answering using natural language  
- Context-aware responses through vector-based semantic search  
- Specialized in Indian Constitutional Law (Part III)  
- Simple and interactive user interface built with Gradio  
- Accurate and reliable retrieval-based answers  

---

## 🔧 Technology Stack

- **LangChain** – RAG pipeline orchestration  
- **FAISS** – Vector database for semantic search  
- **Hugging Face Transformers**  
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`  
  - LLM: `google/flan-t5-large`  
- **Gradio** – Web-based interface  
- **PyPDF** – PDF document processing  

---

## 📄 Dataset

This chatbot is trained on `part3.pdf`, which contains Articles 12–35 of the Indian Constitution covering Fundamental Rights.

---

## 📖 How It Works

1. The user enters a query related to Fundamental Rights.  
2. The system converts the query into vector embeddings.  
3. Relevant document chunks are retrieved using FAISS.  
4. Retrieved context is passed to the language model.  
5. The model generates an accurate and simplified response.  

---

## 📝 Example Questions

You can try asking the chatbot questions such as:

- What is Article 21?  
- What are the six freedoms under Article 19?  
- Explain the Right to Equality.  
- What does Article 17 say?  
- What are the rights related to arrest under Article 22?  
- What are Fundamental Rights?  

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or above  
- pip (Python package manager)

---

### Installation

```bash
# Clone the repository
git clone https://github.com/parijatpalak12/fundamental-rights-bot.git
cd NayayaBot

# Install dependencies
pip install -r requirements.txt

---

## 🌐 Deploy on Hugging Face

You can easily deploy this chatbot on Hugging Face Spaces.

### Steps to Deploy:

1. Go to https://huggingface.co/spaces and create a new Space.
2. Select **Gradio** as the SDK.
3. Upload the following files:
   - `app.py`
   - `requirements.txt`
   - `part3.pdf` (or any PDF you want to use)
4. Click **Create Space**.

Hugging Face will automatically install dependencies and run `app.py`.

Once deployed, your chatbot will be accessible through a public link.

### 📌 Note:
Users can replace `part3.pdf` with their own PDF file to customize the chatbot for different documents.



