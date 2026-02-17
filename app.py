# =========================
# RAG PDF Chatbot - app.py
# =========================

import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS

from transformers import pipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# --------- Load PDF ---------
PDF_PATH = "ml_notes.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()


# --------- Split Text ---------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)
docs = text_splitter.split_documents(documents)


# --------- Embeddings ---------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------- Vector Store ---------
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# --------- LLM (FLAN-T5) ---------
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=200
)
llm = HuggingFacePipeline(pipeline=pipe)


# --------- Prompt ---------
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using ONLY the context below.
    If the answer is not in the context, say "I don't know".

    Context:
    {context}

    Question:
    {question}
    """
)


# --------- RAG Chain ---------
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# --------- Gradio UI ---------
def chat(question):
    return rag_chain.invoke(question)


demo = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(lines=2, placeholder="Ask from the PDF..."),
    outputs="text",
    title="📚 RAG PDF Chatbot",
    description="Ask questions grounded in your PDF using RAG"
)

demo.launch()
