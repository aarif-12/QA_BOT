from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document   # ✅ Import Document for OCR fallback
import gradio as gr
import warnings
import logging
import pytesseract
from pdf2image import convert_from_path
import socket

# -------------------------------
# Logging setup
# -------------------------------
logging.basicConfig(level=logging.INFO)
warnings.warn = lambda *args, **kwargs: None
warnings.filterwarnings('ignore')

# -------------------------------
# OpenRouter API Key
# -------------------------------
OPENROUTER_KEY = ""
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# -------------------------------
# Local embedding model
# -------------------------------
def get_embeddings():
    """Use local HuggingFace embeddings instead of API calls"""
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        logging.error(f"Failed to load local embeddings: {str(e)}")
        raise

# -------------------------------
# LLM using OpenRouter GPT-4o
# -------------------------------
def get_llm():
    return ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        openai_api_base=OPENROUTER_BASE,
        openai_api_key=OPENROUTER_KEY
    )

# -------------------------------
# Document loader with OCR fallback
# -------------------------------
def document_loader(file):
    loader = PyPDFLoader(file.name)
    docs = loader.load()

    # If no text is found in the PDF, fall back to OCR
    if all(not doc.page_content.strip() for doc in docs):
        logging.info("No text found in PDF, trying OCR...")

        # 👇 IMPORTANT: pass the poppler_path for Windows
        pages = convert_from_path(
            file.name,
            poppler_path=r"C:\Users\aarif\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"
        )

        ocr_docs = []
        for i, page in enumerate(pages):
            try:
                text = pytesseract.image_to_string(page)
                ocr_docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": file.name, "page": i + 1}
                    )
                )
                logging.info(f"OCR processed page {i}")
            except Exception as e:
                logging.warning(f"OCR failed on page {i}: {str(e)}")
        return ocr_docs

    return docs

# -------------------------------
# Text splitter
# -------------------------------
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    return splitter.split_documents(data)

# -------------------------------
# Vector DB + Retriever
# -------------------------------
def vector_database(chunks):
    non_empty_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    if not non_empty_chunks:
        raise ValueError("No text found in PDF.")

    texts = [chunk.page_content for chunk in non_empty_chunks]
    embedding_model = get_embeddings()

    embeddings = []
    valid_texts = []
    for i, text in enumerate(texts):
        try:
            emb_list = embedding_model.embed_documents([text])
            if emb_list:
                embeddings.append(emb_list[0])
                valid_texts.append(text)
            else:
                logging.warning(f"Chunk {i} returned invalid embedding. Skipping.")
        except Exception as e:
            logging.warning(f"Chunk {i} embedding failed: {str(e)}")

    if not embeddings:
        raise ValueError("No valid embeddings could be generated from the PDF.")

    vectordb = Chroma.from_texts(valid_texts, embedding=embedding_model)
    return vectordb

def retriever(file):
    loaded_docs = document_loader(file)
    chunks = text_splitter(loaded_docs)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()

# -------------------------------
# QA Chain
# -------------------------------
def retriever_qa(file, query):
    try:
        llm = get_llm()
        retriever_obj = retriever(file)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_obj,
            return_source_documents=True
        )
        response = qa.invoke({"query": query})
        return response["result"]
    except Exception as e:
        logging.error(f"QA processing failed: {str(e)}")
        return f"Error: {str(e)}"

# -------------------------------
# Gradio UI
# -------------------------------
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here..."),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="RAG PDF Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

# Launch the app
def find_free_port(start_port=7860):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    return 8080  # fallback

free_port = find_free_port()
print(f"Starting server on port {free_port}")
rag_application.launch(server_name="0.0.0.0", server_port=free_port)