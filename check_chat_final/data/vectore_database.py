import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")


# Define the directory where you want to save the vector store
save_path = r"D:\langchain_ai_bot\data\vectorstore"  # Change this to your desired path

# Ensure the directory exists
os.makedirs(save_path, exist_ok=True)
# Define function to create the vector store and save it
def create_and_save_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load the CSV containing your documents
    df = pd.read_csv(r'.\cleaned_combined.csv')
    df = df.dropna(subset=['combined'])

    # Prepare the documents (assuming there's a 'combined' column in your CSV)
    documents = []
    for row in df.itertuples():
        text_data = getattr(row, 'combined', '')
        document_ch = Document(page_content=text_data)
        documents.append(document_ch)

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents[:5000])

    # Create vector store (FAISS in this case)
    vectorstore = FAISS.from_documents(splits, embeddings)

    # Save the vector store to disk
    vectorstore.save_local(save_path)

    print("Vector store saved to disk.")

# Run the function to create and save the vector store
create_and_save_vectorstore()
