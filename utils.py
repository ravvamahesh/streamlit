import io
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def extract_text_from_pdf(pdf):
    """Extract text from a PDF file."""
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf.read()))
        text = "\n\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def initialize_vector_index(text, api_key):
    """Initialize FAISS vector index from text."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        texts = text_splitter.split_text(text)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        vector_index = FAISS.from_texts(texts, embeddings).as_retriever()
        return vector_index
    except Exception as e:
        raise Exception(f"Error initializing vector index: {str(e)}")

def get_response(question, vector_index, api_key):
    """Get response from Gemini model using RAG."""
    try:
        prompt_template = """
        Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say, "The answer is not available in the context." Do not provide incorrect information.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.3,
            google_api_key=api_key,
            api_version="v1beta"
        )
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        docs = vector_index.get_relevant_documents(question)
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return response['output_text']
    except Exception as e:
        return f"Error generating response: {str(e)}"