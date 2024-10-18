
# Chat with PDF

"Chat with PDF" is a web application that allows users to interactively query and chat about the content of PDF documents. Utilizing advanced natural language processing (NLP) techniques and generative AI, this application makes it easy to extract insights from documents in a conversational manner.
![App Screenshot](https://github.com/Rishi-Sutar/PDF_RAG_AGENT/blob/master/pdfrag.png)


## Features

- **PDF Upload:** Users can upload PDF files for processing.
- **Text Extraction:** The application extracts text from uploaded PDFs.
- **Chunking:** The extracted text is split into manageable chunks for efficient querying.
- **Conversational Interface:** Users can ask questions and receive context-aware responses.
- **User-Friendly UI:** Built with Streamlit for an intuitive and responsive experience.

## Installation

To set up Healwise-AI on your local machine using Conda, follow these steps:

- Clone the repository:

```bash
git clone https://github.com/yourusername/chat-with-pdf.git
cd chat-with-pdf
```

- Create and activate a Conda environment:

```bash
conda create -n chat-with-pdf python=3.9
conda activate chat-with-pdf
```

- Install required packages:

```bash
pip install -r requirements.txt
```


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

- Set up environment variables. Create a .env file in the root directory and add your Google API key:
    
```bash
GOOGLE_API_KEY=your_api_key_here
```
## Usage

#### Data Ingestion, Transformation, and Model Training

- Run the application:

```bash
streamlit run app.py
```
- Open your browser and go to http://localhost:8501.

- Upload a PDF file and ask questions about its content!

## Technologies Used

- Python
- Streamlit
- Langchain
- Google Generative AI
- FAISS (Facebook AI Similarity Search)
- dotenv for environment variable management
- Logging for error handling
## License

This project is licensed under the GPU License. See the LICENSE file for details.

