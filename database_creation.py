from pinecone.grpc import PineconeGRPC as Pinecone
import streamlit as st
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pandas as pd
from langchain_core.documents import Document

embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
index_name = "product"
pc = Pinecone(api_key=pinecone_api_key)

if index_name not in pc.list_indexes().names():
        st.info(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=768,  # Dimension for text-embedding-004
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        

DATASET_PATH = '.\\data\\fully_automated_dataset.csv'
# 1. Load and process the source data
try:
    df = pd.read_csv(DATASET_PATH)
    # Get unique contexts (definition + rubric) to avoid duplicates
    unique_contexts = df['context'].unique().tolist()
    # Convert each context string into a LangChain Document object
    documents = [Document(page_content=context) for context in unique_contexts]
    print(f"✅ Loaded and processed {len(documents)} unique facet definitions.")
except FileNotFoundError:
    print(f"❌ Error: Dataset file not found at '{DATASET_PATH}'.")
    exit()
    
print("Embedding documents and uploading to Pinecone... (This may take a few minutes)")
vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=index_name
)

print("\n🎉 Indexing complete! Your RAG knowledge base is ready.")