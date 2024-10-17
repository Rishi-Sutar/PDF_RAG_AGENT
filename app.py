import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import sys
import tempfile
from src.exception import CustomException
from src.logger import logging

from src.htmlTemplates import css, bot_template, user_template

# Load environment variables
load_dotenv()
gemini_api_key = os.environ.get('google_api_key')

def pdf_loader(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        logging.info("PDF Loaded")
        return docs
    except Exception as e:
        raise CustomException(e, sys)

def get_pdf_chunk(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    
    text_chunks = text_splitter.split_documents(docs)
    logging.info("PDF splited into chunks")
    return text_chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    logging.info("Vectorstore created")
    return vector_store

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            
def disable():
    st.session_state["disabled"] = False

def main():
    st.set_page_config(
        page_title="Chat with PDF",
        page_icon=":books:"
    )
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # Set disabled to True initially
    if "disabled" not in st.session_state:
        st.session_state["disabled"] = True

    with st.sidebar:
        st.subheader("Your PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            if st.button("Process"):
                with st.spinner("Processing..."):
                    try:
                        docs = pdf_loader(uploaded_file)
                        
                        if not docs:
                            st.error("No text found in the PDF.")
                            return

                        text_chunks = get_pdf_chunk(docs)

                        vector_store = get_vectorstore(text_chunks)

                        st.session_state.conversation = get_conversation_chain(vector_store)
                        
                        # Enable text input after processing
                        st.session_state["disabled"] = False
                            
                    except CustomException as e:
                        st.error(f"An error occurred: {e}")


    st.header("Chat with PDF :books:")
    user_question = st.text_input("Ask a question about your documents:",
                                    disabled=st.session_state.disabled, 
                                    on_change=disable
                                )
    if user_question:
        handle_userinput(user_question)
    
if __name__ == "__main__":
    main()
