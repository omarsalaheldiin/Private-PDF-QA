import os
import warnings
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

    

# Suppress warnings for a cleaner UI experience
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
CHROMA_PATH = "chroma_db"
# High-performance open-source embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
# Local Open Source LLM (Requires Ollama running locally)
LLM_MODEL = "llama3" 

class RAGSystem:
    def __init__(self):
        # Professional setup: Initialize models once to save memory
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.llm = ChatOllama(model=LLM_MODEL)
        self.vector_db = None

    def process_document(self, file_path):
        """Loads PDF, splits text, and creates a persistent vector database."""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Professional chunking strategy
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create/Update persistent vector store
            self.vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=CHROMA_PATH
            )
            return "✅ Document indexed successfully! You can now ask questions."
        except Exception as e:
            return f"❌ Error processing document: {str(e)}"

    def ask_question(self, query, history):
        if not self.vector_db:
            return "⚠️ Please upload and process a PDF document first."
        
        # Define a professional prompt for the AI
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Create the modern chain
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(self.vector_db.as_retriever(), question_answer_chain)

        # Execute
        response = rag_chain.invoke({"input": query})
        return response["answer"]

# Initialize the logic
rag_sys = RAGSystem()

# --- PROFESSIONAL GRADIO UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Private AI Document Assistant")
    gr.Markdown("Analyze your PDFs locally and privately using RAG (Llama 3 + ChromaDB).")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="1. Upload PDF", file_types=['.pdf'])
            process_btn = gr.Button("Build Knowledge Base", variant="primary")
            status_box = gr.Textbox(label="System Status", interactive=False)
        
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=rag_sys.ask_question,
                title="2. Chat with your Data"
            )

    # Linking buttons to functions
    process_btn.click(rag_sys.process_document, inputs=[file_input], outputs=[status_box])

if __name__ == "__main__":
    demo.launch(server_name='127.0.0.1', server_port=7860)