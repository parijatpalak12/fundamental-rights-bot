import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os

# Ensure the PDF is available
# In a Hugging Face Space, you would typically upload your PDF directly
# or ensure it's in a subfolder. For this example, we assume it's at the root.
PDF_PATH = "part3.pdf" # Make sure your PDF file is named part3.pdf in the root of your Space

# 1. Load PDF Document
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# 2. Improved Text Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len
)
docs = text_splitter.split_documents(documents)

# 3. Create Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Enhanced Vector Store & Retrieval
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3
    }
)

# 5. Improved Language Model
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_new_tokens=512,
    temperature=0.3,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.2
)
llm = HuggingFacePipeline(pipeline=pipe)

# 6. Context Formatter
def format_docs(docs):
    """Format retrieved documents with clear structure"""
    formatted_chunks = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip()
        content = ' '.join(content.split())
        formatted_chunks.append(f"[Context {i}]\n{content}\n")
    return "\n".join(formatted_chunks)

# 7. Improved Prompt Template
prompt = ChatPromptTemplate.from_template(
    """You are a legal assistant specializing in Indian Constitutional Law, specifically Part III on Fundamental Rights.\n\nUse the following context from the Constitution of India to answer the question accurately and thoroughly.\n\nContext:\n{context}\n\nQuestion: {question}\n\nInstructions for your answer:\n- Provide complete and detailed information\n- When listing items (like freedoms or rights), number them clearly (1., 2., 3., etc.)\n- Quote directly from the Constitution when relevant\n- If the question asks about a specific article, include the article text\n- Be comprehensive - don't truncate your response\n- If the information is not in the context, state that clearly\n\nDetailed Answer:"""
)

# 8. Build the RAG Chain
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 9. Create Enhanced Chat Function
def chat(question):
    """Enhanced chat function with error handling"""
    try:
        if not question or not question.strip():
            return "Please enter a question."
        question = question.strip()
        response = rag_chain.invoke(question)
        response = response.strip()
        if len(response) < 50:
            response += "\n\n(Note: If this answer seems incomplete, try rephrasing your question or asking for more details.)"
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}\n\nPlease try rephrasing your question."

# 10. Launch Gradio Interface
interface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Ask about Fundamental Rights (e.g., 'What are the freedoms under Article 19?')...",
        label="Your Question"
    ),
    outputs=gr.Textbox(
        label="Answer",
        lines=10
    ),
    title="📚 Fundamental Rights Chatbot (Enhanced)",
    description="""
    Ask questions about Part III of the Indian Constitution (Fundamental Rights).\n\n    **Tips for better answers:**\n    - Be specific about article numbers if you know them\n    - Ask about rights, freedoms, or protections\n    - For lists, explicitly ask for "all freedoms" or "all rights"\n\n    **Example questions:**\n    - What is Article 21?\n    - List all six freedoms in Article 19\n    - What is Article 17?\n    - Explain the Right to Equality\n    - What does Article 22 say about arrests?\n    """,
    examples=[
        ["What is Article 21?"],
        ["What are the six freedoms guaranteed by Article 19?"],
        ["What is Article 17?"],
        ["Explain the Right to Equality"],
        ["What does Article 22 say about arrests?"]
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    interface.launch(debug=True)
