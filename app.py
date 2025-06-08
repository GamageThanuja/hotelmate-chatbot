# Updated imports to fix deprecation warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import streamlit.components.v1 as components

# Import Groq
from groq import Groq
from langchain_groq import ChatGroq

# Configure Streamlit to allow iframe embedding
st.set_page_config(
    page_title="HotelMate AI",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add CSS to hide Streamlit branding and make it iframe-friendly
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Remove padding for iframe */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Hide deploy button */
    .stDeployButton {display:none;}
    
    /* Iframe-friendly styling */
    .stApp {
        background: transparent;
    }
    
    /* Chat styling */
    .stChatMessage {
        margin-bottom: 1rem;
    }
    .stSpinner {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Create title with app icon inline
import base64

# Function to convert image to base64
def get_base64_of_image(path):
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Get base64 of the image
img_base64 = get_base64_of_image("app-icon.png")

if img_base64:
    st.markdown(
        f'<h1 style="display: flex; align-items: center;"><img src="data:image/png;base64,{img_base64}" width="32" style="margin-right: 8px;"> <span style="margin-top: 3px;">HotelMate AI</span></h1>',
        unsafe_allow_html=True
    )
else:
    st.title('🤖 HotelMate AI')

# Initialize Groq
groq_api_key = st.secrets["GROQ_API_KEY"]

if groq_api_key:
    client = Groq(api_key=groq_api_key)
    # Initialize LangChain Groq LLM for PDF processing
    llm = ChatGroq(
        model="llama3-8b-8192",
        groq_api_key=groq_api_key,
        temperature=0.7
    )

@st.cache_resource
def load_pdf():
    pdf_name = 'hotemate.pdf'
    loaders = [PyPDFLoader(pdf_name)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    ).from_loaders(loaders)
    return index

# Load PDF index
try:
    index = load_pdf()
    pdf_loaded = True
    
    # Create QA chain with better prompt template
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 3}),
        input_key='question',
        return_source_documents=True
    )
except Exception as e:
    st.error(f"❌ Error loading PDF: {e}")
    pdf_loaded = False

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Add welcome message if no messages exist
if not st.session_state.messages:
    welcome_msg = "How can I assist you today?"
    st.session_state.messages.append({'role': 'assistant', 'content': welcome_msg})

# Display chat history
for message in st.session_state.messages: 
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Chat input
if prompt := st.chat_input("Ask me anything about our hotel services..."):
    # Add user message to chat history
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate AI response
    if groq_api_key and pdf_loaded and 'chain' in locals():
        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Use PDF knowledge for response
                    result = chain({"question": prompt})
                    response = result['result']
                    
                    # Clean up the response to make it more natural
                    if response.startswith("According to the provided context, "):
                        response = response.replace("According to the provided context, ", "")
                    if response.startswith("According to the context, "):
                        response = response.replace("According to the context, ", "")
                    
                    # Make first letter uppercase if it's lowercase after cleaning
                    if response and response[0].islower():
                        response = response[0].upper() + response[1:]
                    
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({'role': 'assistant', 'content': response})
                    
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            error_msg = "I apologize, but I'm having trouble processing your request right now. Please try again or rephrase your question."
            st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
    else:
        error_msg = "Sorry, I'm not properly configured. Please check the API key and PDF file."
        with st.chat_message("assistant"):
            st.markdown(error_msg)
        st.session_state.messages.append({'role': 'assistant', 'content': error_msg})