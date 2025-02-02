from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Extract data from PDFs and TXT files
def load_pdf_file(data_path):
    pdf_loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader)

    pdf_documents = pdf_loader.load()
    txt_documents = txt_loader.load()

    return pdf_documents + txt_documents  # Combine both types of documents

# Splits the extracted text into smaller chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Download the embeddings from Hugging Face
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  # This model returns 384 dimensions
    return embeddings
