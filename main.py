import os
import pandas as pd
import pypdf
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# --- App Configuration ---
st.set_page_config(page_title="Hotel Strategy AI", page_icon="🏨", layout="wide")
st.title("🏨 Premium Hotel Strategy AI")
st.write("This tool synthesizes expert research and customer reviews to generate strategic recommendations.")

# --- API Key Handling for Sharing ---
# This block correctly loads the key for both local and deployed versions.
try:
    # First, try to get the key from Streamlit's secrets manager
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except:
    # If that fails (i.e., we're running locally), load from the .env file
    load_dotenv()

# --- Function to Load Data ---
@st.cache_data(show_spinner="Loading and processing knowledge base...")
def load_and_process_data():
    knowledge_base_texts = []
    research_folder = "research"
    if os.path.exists(research_folder):
        for filename in os.listdir(research_folder):
            file_path = os.path.join(research_folder, filename)
            if filename.endswith(".txt"):
                with open(file_path, "r", encoding='utf-8') as f:
                    knowledge_base_texts.append(f.read())
            elif filename.endswith(".pdf"):
                try:
                    pdf_reader = pypdf.PdfReader(file_path)
                    pdf_text = "".join(page.extract_text() for page in pdf_reader.pages)
                    knowledge_base_texts.append(pdf_text)
                except Exception: pass
    try:
        reviews_df = pd.read_csv("reviews.csv")
    except FileNotFoundError:
        return None, None
    combined_knowledge = "\n\n---\n\n".join(knowledge_base_texts)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    texts = text_splitter.split_text(combined_knowledge)
    return texts, reviews_df

# --- Main App Logic ---
texts, reviews_df = load_and_process_data()
if texts and reviews_df is not None:
    try:
        st.success("Knowledge base and reviews loaded successfully!")
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_texts(texts, embeddings)
        llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.5)
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=compression_retriever)
        
        st.header("Ask a Strategic Question")
        hotel_list = ["All Hotels"] + reviews_df['Hotel'].unique().tolist()
        selected_hotel = st.selectbox("Select a hotel to analyze:", hotel_list)
        default_query = "Based on the provided research and customer reviews, what are the top 3 recommendations for improving the premium check-in experience?"
        user_query = st.text_area("Your Question:", value=default_query, height=150)

        if st.button("Generate Strategy", type="primary"):
            with st.spinner("AI is thinking..."):
                if selected_hotel == "All Hotels":
                    relevant_reviews = reviews_df['Review'].tolist()
                else:
                    relevant_reviews = reviews_df[reviews_df['Hotel'] == selected_hotel]['Review'].tolist()
                all_reviews_text = "\n\n---\n\n".join(relevant_reviews)
                final_prompt = f"You are a top-tier hospitality strategy consultant. Your Goal: Answer the user's question by synthesizing expert research with real customer feedback for {selected_hotel}. User's Question: {user_query} Supporting Customer Reviews: {all_reviews_text}"
                
                result = qa_chain.invoke(final_prompt)
                st.subheader("AI Strategic Recommendation")
                st.markdown(result['result'])

    except Exception as e:
        # Catch the OpenAIError specifically if it still occurs
        if "api_key" in str(e):
            st.error("Authentication Error: Could not find the OpenAI API key. Please make sure it is set correctly in the Streamlit app settings under 'Secrets'.")
        else:
            st.error(f"An unexpected error occurred: {e}")
else:
    st.error("Error: Could not find research papers or 'reviews.csv'. Please check your files on GitHub.")