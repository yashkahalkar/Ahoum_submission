# app.py

import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Functions (cached for performance) ---

@st.cache_resource
def get_vectorstore():
    """Initializes the Pinecone Vector Store from an existing index."""
    # Use the same embedding model as in your indexing script
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    # The user changed the index name to "product"
    index_name = "product"
    
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    return vectorstore

@st.cache_resource
def get_llm():
    """Initializes and returns the local LLM via Ollama."""
    # Ensure Ollama is running with the 'phi3' model
    return ChatOllama(model="phi3", temperature=0)

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🤖 RAG-Based Conversation Scorer (LangChain & Pinecone)")
st.write("This app uses a local LLM (Phi-3) to score conversations based on rubrics retrieved from a Pinecone vector database.")

# Initialize components
try:
    vectorstore = get_vectorstore()
    llm = get_llm()
    st.success("✅ Connected to Pinecone and local LLM successfully!")
except Exception as e:
    st.error(f"❌ Failed to initialize components. Please check your API keys and ensure Ollama is running. Error: {e}")
    st.stop()

# User Inputs
col1, col2 = st.columns(2)
with col1:
    facet_name = st.text_input(
        "**1. Enter the Facet to Score**",
        value="Assertiveness",
        help="The system will search for the rubric related to this facet."
    )
with col2:
    conversation_turn = st.text_area(
        "**2. Enter the Conversation Turn to Analyze**",
        value="We need to consider other options before moving forward with this plan.",
        height=150
    )

if st.button("Score Conversation", type="primary", use_container_width=True):
    if not facet_name or not conversation_turn:
        st.warning("Please provide both a facet and a conversation turn.")
    else:
        with st.spinner("Retrieving rubric and generating score..."):
            try:
                # 1. Create a retriever from the vector store
                retriever = vectorstore.as_retriever()

                # 2. Define the prompt template
                prompt_template = ChatPromptTemplate.from_template(
                    """
                    **Task:** Score the following conversation turn on a scale of 1 to 5 based ONLY on the provided rubric.

                    **Rubric:**
                    {context}

                    **Conversation Turn:**
                    "{conversation}"

                    **Instructions:**
                    - Read the rubric carefully.
                    - Evaluate how well the conversation turn matches the rubric's definitions.
                    - Your response MUST be a single number from 1 to 5. Do not add any other text, explanation, or punctuation.

                    **Score (1-5):**
                    """
                )
                
                # Helper function to format retrieved documents
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                # 3. Create the RAG chain using LangChain Expression Language (LCEL)
                rag_chain = (
                    # The `RunnablePassthrough` allows the conversation to be passed through the chain
                    {"context": retriever | format_docs, "conversation": RunnablePassthrough()}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )

                # 4. Invoke the chain with the conversation turn as input
                score = rag_chain.invoke(conversation_turn)
                
                # Retrieve the context again just for display purposes
                retrieved_docs = retriever.get_relevant_documents(facet_name)
                
                st.divider()
                st.subheader("Results")
                
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    st.metric(f"Score for '{facet_name}'", score.strip())
                with res_col2:
                    st.info("**Retrieved Rubric Used for Scoring:**")
                    # Display the content of the first retrieved document
                    if retrieved_docs:
                        st.text(retrieved_docs[0].page_content)
                    else:
                        st.warning("No relevant rubric was found for the given facet.")

            except Exception as e:
                st.error(f"An error occurred during scoring: {e}")