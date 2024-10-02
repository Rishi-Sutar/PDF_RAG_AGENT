from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()   

class DocumentIndexerConfig():
    self.document = None

class DocumentIndexer(DocumentIndexerConfig):
    self.document