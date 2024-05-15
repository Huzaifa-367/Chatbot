import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import tempfile
from gtts import gTTS
import os

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=api_key, model_name="sentence-transformers/all-MiniLM-l6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGroq(temperature=0, groq_api_key=os.environ["groq_api_key"], model_name="llama3-8b-8192")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain
    
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_filename = audio_file.name
    tts.save(temp_filename)
    st.audio(temp_filename, format='audio/mp3')
    os.remove(temp_filename)

def user_input(user_question, api_key):
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=api_key, model_name="sentence-transformers/all-MiniLM-l6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Replies:")
    if isinstance(response["output_text"], str):
        response_list = [response["output_text"]]
    else:
        response_list = response["output_text"]
    for text in response_list:
        st.write(text)
        # Convert text to speech for each response
        text_to_speech(text)

def main():
    st.set_page_config(layout="wide")
    st.header("Chat with DOCS")
    st.markdown("<h1 style='font-size:20px;'>ChatBot by Muhammad Huzaifa</h1>", unsafe_allow_html=True)
    api_key = st.secrets["inference_api_key"]

    # Sidebar column for file upload
    with st.sidebar:
        st.header("Chat with PDF")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])

    # Main column for displaying extracted text and user interaction
    col1, col2 = st.columns([1, 2])
    raw_text = None
    if pdf_docs is None:
        with col1:
            st.write("Please upload a document first to proceed.")
    if pdf_docs:
        with col1:
            if st.button("Submit"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, api_key)
                    st.success("Processing Complete")
            user_question = st.text_input("Ask a question from the Docs")
            if user_question:
                raw_text = get_pdf_text(pdf_docs)
                user_input(user_question, api_key)
 
    # Display extracted text if available
    if raw_text is not None:
        with col2:
            st.subheader("Extracted Text from PDF:")
            st.text(raw_text)

if __name__ == "__main__":
    main()
