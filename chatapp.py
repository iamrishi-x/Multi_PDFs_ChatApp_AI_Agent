import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import io
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_nomic import NomicEmbeddings
from langchain_community.vectorstores import FAISS

# streamlit run chatapp.py

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
os.environ["NOMIC_API_KEY"] = os.getenv("NOMIC_API_KEY")


def get_text_chunks(text,ids,file_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    ids_current = [f"{file_name}.{l}" for l in range(len(chunks))]
    ids+=ids_current
    return chunks,ids

def get_pdf_text(pdf_docs):
    ids = []
    chunks_all = []
    for pdf in pdf_docs:
        # Convert the UploadedFile to a file-like object
        pdf_file = io.BytesIO(pdf.getvalue())
        pdf_reader = PdfReader(pdf_file)
        print(f"Processing file - {pdf.name}")
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        chunks, ids = get_text_chunks(text, ids, pdf.name)
        chunks_all += chunks
    return chunks_all, ids

def get_vector_store(embeddings_api,text_chunks,ids):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings_api,ids=ids)
    vector_store.save_local("faiss_index")
    #return vector_store

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question,embeddings):
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    
    print(response)
    st.write(f"Reply: {response['output_text']}")
    st.subheader("Response came from: ", divider=True)
    data = []
    for doc in docs:
        data.append({"file": doc.id, "content": doc.page_content})
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

def main():
    #embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
    st.set_page_config("Multi PDF Chatbot", page_icon = ":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")

    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")

    if user_question:
        user_input(user_question,embeddings)

    with st.sidebar:  #---only with Submit/Process is hit---#

        #st.image("img/Robot.jpg")
        st.write("---")
        
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
        # pdfs = [file.name for file in pdf_docs]
        print("*"*30+f"pdfs - {pdf_docs}"+"*"*30)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."): # user friendly message.
                chunks_all,ids = get_pdf_text(pdf_docs=pdf_docs) # get the pdf text
                get_vector_store(embeddings,chunks_all,ids) # create vector store
                st.success("Done")
        
        st.write("---")
        st.write("AI App created by @ Rishi Bagul")  


    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/iamrishi-x" target="_blank">Rishi Bagul</a> | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()