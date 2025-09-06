# main.py

import asyncio
import os
import time
from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.ui import pdf_uploader
from app.pdf_utils import extract_text_from_pdf
from app.vectorstore_utils import create_faiss_index, retrieve_relevant_docs
from app.chat_utils import generate_response  # updated for Gemini

# ---------------------------
# Fix for Windows + Streamlit asyncio
# ---------------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="MediChat Pro - Medical Document Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Your existing CSS here */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 20px;
    background-color: #f5f5f5;  /* reverted to original light color */
    border-bottom: 1px solid #ddd;
    font-family: sans-serif;
}
.navbar-left {
    font-weight: bold;
    font-size: 1.1rem;
    color: #555
}
.navbar-right {
    font-size: 0.9rem;
    color: #555;
    margin-right: 20px; /* space from the right edge */
}
</style>

<div class="navbar">
    <div class="navbar-left">
        <span class="icon">🩺</span> MediChat Pro - Medical Document Assistant
    </div>
    <div class="navbar-right">
        Powered by Gemini 2.5 Pro
    </div>
</div>
<div class="main">
""", unsafe_allow_html=True)




# ---------------------------
# Session State
# ---------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

# ---------------------------
# Sidebar: Upload & Process Docs
# ---------------------------
with st.sidebar:
    st.markdown("### 📂 Document Upload")
    uploaded_files = pdf_uploader()

    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} document(s) uploaded")
        if st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing your medical documents..."):
                all_texts = []
                for f in uploaded_files:
                    text = extract_text_from_pdf(f)
                    if text.strip():
                        all_texts.append(text)
                    else:
                        st.warning(f"⚠️ No text extracted from {f.name}")

                if all_texts:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = []
                    for text in all_texts:
                        chunks.extend(splitter.split_text(text))
                    st.session_state.vectorstore = create_faiss_index(chunks)
                    st.success("✅ Documents processed successfully!")
                    st.balloons()
                else:
                    st.error("❌ No valid text found in uploaded documents.")

    if st.button("🧹 Clear Chat"):
        st.session_state["messages"] = []
        st.success("Chat history cleared!")
        st.rerun()

# ---------------------------
# Main Chat Interface
# ---------------------------
st.markdown("<h1 style='color:#009688; text-align:center; margin-top:20px;'>🩺 MediChat Pro</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center; margin-top:30px;'>💬 Chat with your Medical Documents</h2>", unsafe_allow_html=True)

# Display previous messages
for message in st.session_state.messages:
    role_class = "user" if message["role"] == "user" else "assistant"
    st.markdown(f"<div class='chat-message {role_class}'>{message['content']}<br><small>{message['timestamp']}</small></div>", unsafe_allow_html=True)

# User input
if prompt := st.chat_input("Ask about your medical documents..."):
    timestamp = time.strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
    st.markdown(f"<div class='chat-message user'>{prompt}<br><small>{timestamp}</small></div>", unsafe_allow_html=True)

    if st.session_state.vectorstore:
        with st.chat_message("assistant"):
            with st.spinner("🔎 Searching documents..."):
                # Retrieve relevant docs from FAISS
                relevant_docs = retrieve_relevant_docs(st.session_state.vectorstore, prompt)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                # Build system prompt
                system_prompt = f"""You are MediChat Pro, an intelligent medical document assistant.
Based on the following medical documents, provide accurate and helpful answers.
If the information is not in the documents, clearly state that.
When answering, take help from the LLM and give a detailed medical explanation.

Medical Documents:
{context}

User Question: {prompt}

Answer:"""

                # Generate response using new Gemini SDK
                response = generate_response(system_prompt)

            # Display assistant response
            st.markdown(f"<div class='chat-message assistant'>{response}<br><small>{timestamp}</small></div>", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": timestamp})
    else:
        st.warning("⚠️ Please upload and process documents first!")

st.markdown("</div>", unsafe_allow_html=True)






# --- Function to Inject CSS ---
def load_css():
    """Injects custom CSS to style the Streamlit app."""
    st.markdown(
        """
        <style>
        /* Custom footer */
        .custom-footer {
            position: fixed;
            right: 10px;
            bottom: 10px;
            width: auto;
            color: #a0a0a0; /* Light grey text */
            text-align: right;
            font-size: 12px;
            font-family: 'Inter', sans-serif;
            background-color: transparent;
            z-index: 999; /* Ensure it stays on top of other elements */
        }
        .custom-footer a {
            color: #b0b0e0; /* Light purple link */
            text-decoration: none;
        }
        .custom-footer a:hover {
            text-decoration: underline;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- Call the function to load the CSS ---
load_css()

# --- Create the Footer HTML ---
st.markdown(
    """
    <div class="custom-footer">
        Designed & Developed by <a href="#" target="_blank">Pramod Kumar</a>
    </div>
    """,
    unsafe_allow_html=True,
)


































# import asyncio

# # Fix for Windows + Streamlit
# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())



# import os
# from dotenv import load_dotenv

# # Load environment variables from .env
# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# import streamlit as st 
# import time
# from app.ui import pdf_uploader
# from app.pdf_utils import extract_text_from_pdf
# from app.vectorstore_utils import create_faiss_index, retrieve_relevant_docs  
# from app.chat_utils import get_chat_model, ask_chat_model
# from langchain.text_splitter import RecursiveCharacterTextSplitter


# # -----------------------------------
# # Page Config
# # -----------------------------------
# st.set_page_config(
#     page_title="MediChat Pro - Medical Document Assistant",
#     page_icon="🩺",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Sidebar CSS Styling
# st.markdown(
#     """
#     <style>
#     /* Sidebar background */
#     section[data-testid="stSidebar"] {
#         background-color: #f5f7fa;  /* light gray */
#         color: #000000;  /* black text */
#         padding: 20px;
#     }

#     /* Sidebar headers */
#     section[data-testid="stSidebar"] h1,
#     section[data-testid="stSidebar"] h2,
#     section[data-testid="stSidebar"] h3 {
#         color: #1E3A8A;  /* navy blue headings */
#     }

#     /* Sidebar text */
#     section[data-testid="stSidebar"] p {
#         color: #000000;  /* normal text black */
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


# # -----------------------------------
# # Custom CSS + Navbar
# # -----------------------------------
# st.markdown("""
# <style>
# /* ===== Fixed Top Navbar ===== */
# .navbar { position: fixed; top: 0; left: 0; right: 0; height: 60px;
#     background-color: #009688; display: flex; align-items: center;
#     justify-content: space-between; padding: 0 1.5rem; color: white;
#     font-family: "Segoe UI", sans-serif; font-size: 1.1rem; font-weight: bold;
#     z-index: 999; box-shadow: 0 2px 6px rgba(0,0,0,0.15); }
# .navbar-left { display: flex; align-items: center; }
# .navbar-left .icon { font-size: 1.5rem; margin-right: 0.6rem; }
# .navbar-right { display: flex; align-items: center; }
# .navbar-right .profile { width: 34px; height: 34px; border-radius: 50%;
#     background-color: #ffffff33; margin-right: 1rem; display: flex;
#     align-items: center; justify-content: center; font-size: 1.2rem; }
# .navbar-right .logout-btn { background-color: #ff4b4b; color: white;
#     padding: 0.4rem 0.9rem; border-radius: 0.4rem; text-decoration: none;
#     font-size: 0.9rem; font-weight: 500; transition: background 0.2s ease-in-out; }
# .navbar-right .logout-btn:hover { background-color: #ff3333; }
# .main { margin-top: 70px; }
# [data-testid="stSidebar"] { background-color: #ffffff; border-right: 2px solid #e6f0f5; }
# .chat-message { padding: 1rem; border-radius: 0.6rem; margin-bottom: 1rem;
#     display: flex; flex-direction: column; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
# .chat-message.user { background-color: #2b313e; color: white; }
# .chat-message.assistant { background-color: #e8f7f3; color: #00332e; }
# .stButton > button { background-color: #009688; color: white; border-radius: 0.5rem;
#     border: none; padding: 0.5rem 1.2rem; font-weight: 600;
#     transition: all 0.2s ease-in-out; }
# .stButton > button:hover { background-color: #00796b; transform: scale(1.02); }
# .upload-section { margin-top: 1rem; padding: 1rem; border: 2px dashed #009688;
#     border-radius: 0.5rem; background-color: #f0faf9; text-align: center;
#     color: #004d40; font-style: italic; }
# </style>

# <!-- Navbar HTML -->
# <div class="navbar">
#     <div class="navbar-left">
#         <span class="icon">🩺</span> MediChat Pro - Medical Document Assistant
#     </div>
#     <div class="navbar-right">
#         <div class="profile">👤</div>
#         <a href="#" class="logout-btn">Logout</a>
#     </div>
# </div>
# <div class="main">
# """, unsafe_allow_html=True)

# # -----------------------------------
# # Initialize session state
# # -----------------------------------
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# if "vectorstore" not in st.session_state:
#     st.session_state["vectorstore"] = None

# if "chat_model" not in st.session_state:
#     st.session_state["chat_model"] = None





# # ✅ Custom CSS for Sidebar Background (Light Greyish)
# st.markdown(
#     """
#     <style>
#     /* Sidebar background */
#     section[data-testid="stSidebar"] {
#         background-color: #e5e7eb !important;  /* light grey background */
#         color: #111827 !important;             /* dark text for contrast */
#     }

#     /* Sidebar text */
#     section[data-testid="stSidebar"] .css-1d391kg, 
#     section[data-testid="stSidebar"] .css-1v3fvcr {
#         color: #111827 !important;             /* dark grey/black text */
#     }

#     /* File uploader button (Browse files) */
#     .stFileUploader label div div {
#         color: #ffffff !important;             /* white font */
#         background-color: #6b7280 !important;  /* medium grey */
#         border-radius: 10px;
#         padding: 8px 16px;
#         font-weight: 600;
#         font-size: 15px;
#         text-align: center;
#         cursor: pointer;
#         transition: all 0.3s ease;
#     }

#     /* Hover effect for Browse files */
#     .stFileUploader label div div:hover {
#         background-color: #4b5563 !important;  /* darker grey on hover */
#         color: #f9fafb !important;
#         transform: scale(1.03);
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )




# # -----------------------------------
# # Sidebar for Document Upload
# # -----------------------------------
# with st.sidebar:
#     st.markdown("### 📂 Document Upload")
#     st.markdown("Upload your medical documents to start chatting!")

#     uploaded_files = pdf_uploader()

#     if uploaded_files:
#         st.success(f"📄 {len(uploaded_files)} document(s) uploaded")

#         # Process documents button
#         if st.button("🚀 Process Documents", type="primary"): 
#             with st.spinner("Processing your medical documents..."):
#                 # Extract text from all PDFs
#                 all_texts = []
#                 for file in uploaded_files:
#                     text = extract_text_from_pdf(file)
#                     if text.strip():
#                         all_texts.append(text)
#                     else:
#                         st.warning(f"⚠️ No text extracted from {file.name}")

#                 # Split texts into chunks
#                 text_splitter = RecursiveCharacterTextSplitter(
#                     chunk_size=1000,
#                     chunk_overlap=200,
#                     length_function=len,
#                 )

#                 chunks = []
#                 for text in all_texts:
#                     chunks.extend(text_splitter.split_text(text))

#                 # Create FAISS index
#                 vectorstore = create_faiss_index(chunks)
#                 st.session_state.vectorstore = vectorstore

#                 # Initialize chat model
#                 chat_model = get_chat_model(GOOGLE_API_KEY)
#                 st.session_state.chat_model = chat_model

#                 st.success("✅ Documents processed successfully!")
#                 st.balloons()


#     # Clear Chat button
#     if st.button("🧹 Clear Chat"):
#         st.session_state["messages"] = []
#         st.success("Chat history cleared!")
#         st.rerun()

# # -----------------------------------
# # Main Chat Interface
# # -----------------------------------

# # App Title (below navbar)
# st.markdown(
#     "<h1 style='color:#009688; text-align:center; margin-top:20px;'>🩺 MediChat Pro</h1>",
#     unsafe_allow_html=True
# )

# # Chat Section Heading (centered)
# st.markdown(
#     "<h2 style=' text-align:center; margin-top:30px;'>💬 Chat with your Medical Documents</h2>",
#     unsafe_allow_html=True
# )



# # Show messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#         st.caption(message["timestamp"])

# # Input
# if prompt := st.chat_input("Ask about your medical documents..."):
#     timestamp = time.strftime("%H:%M")
#     st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})

#     with st.chat_message("user"):
#         st.markdown(prompt)
#         st.caption(timestamp)

#     if st.session_state.vectorstore and st.session_state.chat_model:
#         with st.chat_message("assistant"):
#             with st.spinner("🔎 Searching documents..."):
#                 relevant_docs = retrieve_relevant_docs(st.session_state.vectorstore, prompt)
#                 context = "\n\n".join([doc.page_content for doc in relevant_docs])

#                 system_prompt = f"""You are MediChat Pro, an intelligent medical document assistant.
#                 Based on the following medical documents, provide accurate and helpful answers.
#                 If the information is not in the documents, clearly state that.
#                 When answering, take help from the LLM and give a detailed medical explanation.

#                 Medical Documents:
#                 {context}

#                 User Question: {prompt}

#                 Answer:"""

#                 response = ask_chat_model(st.session_state.chat_model, system_prompt)

#             st.markdown(response)
#             st.caption(timestamp)
#             st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": timestamp})
#     else:
#         st.warning("⚠️ Please upload and process documents first!")

# # Close the main div from navbar
# st.markdown("</div>", unsafe_allow_html=True)



















# import streamlit as st 
# from app.ui import pdf_uploader
# from app.pdf_utils import extract_text_from_pdf
# from app.vectorstore_utils import create_faiss_index, retrieve_relevant_docs  
# from app.chat_utils import get_chat_model, ask_chat_model
# from app.config import GOOGLE_API_KEY
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import time 


# st.set_page_config(
#     page_title="MediChat Pro - Medical Document Assistant",
#     page_icon="🩺",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# st.markdown("""
# <style>
# /* ===== Fixed Top Navbar ===== */
# .navbar {
#     position: fixed;
#     top: 0;
#     left: 0;
#     right: 0;
#     height: 60px;
#     background-color: #009688; /* medical teal */
#     display: flex;
#     align-items: center;
#     justify-content: space-between;
#     padding: 0 1.5rem;
#     color: white;
#     font-family: "Segoe UI", sans-serif;
#     font-size: 1.1rem;
#     font-weight: bold;
#     z-index: 999;
#     box-shadow: 0 2px 6px rgba(0,0,0,0.15);
# }

# /* Left section (icon + title) */
# .navbar-left {
#     display: flex;
#     align-items: center;
# }
# .navbar-left .icon {
#     font-size: 1.5rem;
#     margin-right: 0.6rem;
# }

# /* Right section (profile + logout) */
# .navbar-right {
#     display: flex;
#     align-items: center;
# }
# .navbar-right .profile {
#     width: 34px;
#     height: 34px;
#     border-radius: 50%;
#     background-color: #ffffff33;
#     margin-right: 1rem;
#     display: flex;
#     align-items: center;
#     justify-content: center;
#     font-size: 1.2rem;
# }
# .navbar-right .logout-btn {
#     background-color: #ff4b4b;
#     color: white;
#     padding: 0.4rem 0.9rem;
#     border-radius: 0.4rem;
#     text-decoration: none;
#     font-size: 0.9rem;
#     font-weight: 500;
#     transition: background 0.2s ease-in-out;
# }
# .navbar-right .logout-btn:hover {
#     background-color: #ff3333;
# }

# /* Push content below navbar */
# .main {
#     margin-top: 70px;
# }

# /* ===== Sidebar ===== */
# [data-testid="stSidebar"] {
#     background-color: #ffffff;
#     border-right: 2px solid #e6f0f5;
# }

# /* ===== Chat Messages ===== */
# .chat-message {
#     padding: 1rem;
#     border-radius: 0.6rem;
#     margin-bottom: 1rem;
#     display: flex;
#     flex-direction: column;
#     box-shadow: 0 2px 6px rgba(0,0,0,0.08);
# }
# .chat-message.user {
#     background-color: #2b313e;
#     color: white;
# }
# .chat-message.assistant {
#     background-color: #e8f7f3;
#     color: #00332e;
# }

# /* ===== Buttons ===== */
# .stButton > button {
#     background-color: #009688;
#     color: white;
#     border-radius: 0.5rem;
#     border: none;
#     padding: 0.5rem 1.2rem;
#     font-weight: 600;
#     transition: all 0.2s ease-in-out;
# }
# .stButton > button:hover {
#     background-color: #00796b;
#     transform: scale(1.02);
# }

# /* ===== File Upload Section ===== */
# .upload-section {
#     margin-top: 1rem;
#     padding: 1rem;
#     border: 2px dashed #009688;
#     border-radius: 0.5rem;
#     background-color: #f0faf9;
#     text-align: center;
#     color: #004d40;
#     font-style: italic;
# }
# </style>

# <!-- Navbar HTML -->
# <div class="navbar">
#     <div class="navbar-left">
#         <span class="icon">🩺</span> MediChat Pro - Medical Document Assistant
#     </div>
#     <div class="navbar-right">
#         <div class="profile">👤</div>
#         <a href="#" class="logout-btn">Logout</a>
#     </div>
# </div>

# <div class="main">
# """, unsafe_allow_html=True)







# import streamlit as st
# import time
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # ✅ Your helper functions (make sure these exist in your project)
# # - pdf_uploader()
# # - extract_text_from_pdf(file)
# # - create_faiss_index(chunks)
# # - get_chat_model(api_key)
# # - ask_chat_model(chat_model, prompt)
# # - retrieve_relevant_docs(vectorstore, query)

# GOOGLE_API_KEY = "your_api_key_here"  # Replace with your actual key

# # -----------------------------------
# # Initialize session state variables
# # -----------------------------------
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []  # Stores chat history

# if "vectorstore" not in st.session_state:
#     st.session_state["vectorstore"] = None  # Stores FAISS index of documents

# if "chat_model" not in st.session_state:
#     st.session_state["chat_model"] = None  # Stores the initialized LLM model

# # -----------------------------------
# # Sidebar - Document Upload + Clear Chat
# # -----------------------------------
# with st.sidebar:
#     st.markdown("### 📂 Document Upload")
#     st.markdown("Upload your medical documents to start chatting!")

#     uploaded_files = pdf_uploader()

#     if uploaded_files:
#         st.success(f"📄 {len(uploaded_files)} document(s) uploaded")

#         # Process documents when user clicks the button
#         if st.button("🚀 Process Documents", type="primary"): 
#             with st.spinner("Processing your medical documents..."):
#                 # Extract text from all uploaded PDFs
#                 all_texts = []
#                 for file in uploaded_files:
#                     text = extract_text_from_pdf(file)
#                     if text.strip():
#                         all_texts.append(text)
#                     else:
#                         st.warning(f"⚠️ No text extracted from {file.name}")

#                 # Split extracted text into smaller chunks
#                 text_splitter = RecursiveCharacterTextSplitter(
#                     chunk_size=1000,
#                     chunk_overlap=200,
#                     length_function=len,
#                 )
#                 chunks = []
#                 for text in all_texts:
#                     chunks.extend(text_splitter.split_text(text))

#                 # Create FAISS vector store
#                 vectorstore = create_faiss_index(chunks)
#                 st.session_state.vectorstore = vectorstore

#                 # Initialize chat model
#                 chat_model = get_chat_model(GOOGLE_API_KEY)
#                 st.session_state.chat_model = chat_model

#                 st.success("✅ Documents processed successfully!")
#                 st.balloons()

#     # ✅ Add Clear Chat button
#     if st.button("🧹 Clear Chat"):
#         st.session_state["messages"] = []
#         st.success("Chat history cleared!")

# # -----------------------------------
# # Main Chat Interface
# # -----------------------------------
# st.markdown("### 💬 Chat with your Medical Documents")

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#         st.caption(message["timestamp"])

# # Chat input field
# if prompt := st.chat_input("Ask about your medical documents..."):
#     # Save user message
#     timestamp = time.strftime("%H:%M")
#     st.session_state.messages.append({
#         "role": "user",
#         "content": prompt,
#         "timestamp": timestamp
#     })

#     # Display user message
#     with st.chat_message("user"):
#         st.markdown(prompt)
#         st.caption(timestamp)

#     # Generate assistant response
#     if st.session_state.vectorstore and st.session_state.chat_model:
#         with st.chat_message("assistant"):
#             with st.spinner("🔎 Searching documents..."):
#                 # Retrieve relevant documents from FAISS
#                 relevant_docs = retrieve_relevant_docs(st.session_state.vectorstore, prompt)
#                 context = "\n\n".join([doc.page_content for doc in relevant_docs])

#                 # Create system prompt with context
#                 system_prompt = f"""You are MediChat Pro, an intelligent medical document assistant.
#                 Based on the following medical documents, provide accurate and helpful answers.
#                 If the information is not in the documents, clearly state that.
#                 When answering, take help from the LLM and give a detailed medical explanation.

#                 Medical Documents:
#                 {context}

#                 User Question: {prompt}

#                 Answer:"""

#                 # Get response from the chat model
#                 response = ask_chat_model(st.session_state.chat_model, system_prompt)

#             # Display assistant response
#             st.markdown(response)
#             st.caption(timestamp)

#             # Save assistant response in chat history
#             st.session_state.messages.append({
#                 "role": "assistant",
#                 "content": response,
#                 "timestamp": timestamp
#             })
#     else:
#         st.warning("⚠️ Please upload and process documents first!")










# # Sidebar for document upload
# with st.sidebar:
#     st.markdown("### 📂 Document Upload")
#     st.markdown("Upload your medical documents to start chatting!")

#     uploaded_files = pdf_uploader()

#     if uploaded_files:
#         st.success(f"📄 {len(uploaded_files)} document(s) uploaded")

#         # Process documents
#         if st.button("🚀 Process Documents", type="primary"): 
#             with st.spinner("Processing your medical documents..."):
#                 # Extract text from all PDFs
#                 all_texts = []
#                 for file in uploaded_files:
#                     text = extract_text_from_pdf(file)
#                     if text.strip():
#                         all_texts.append(text)
#                     else:
#                         st.warning(f"⚠️ No text extracted from {file.name}")

#                 # Split texts into chunks
#                 text_splitter = RecursiveCharacterTextSplitter(
#                     chunk_size=1000,
#                     chunk_overlap=200,
#                     length_function=len,
#                 )

#                 chunks = []
#                 for text in all_texts:
#                     chunks.extend(text_splitter.split_text(text))

#                 # Create FAISS index
#                 vectorstore = create_faiss_index(chunks)
#                 st.session_state.vectorstore = vectorstore

#                 # Initialize chat model
#                 chat_model = get_chat_model(GOOGLE_API_KEY)
#                 st.session_state.chat_model = chat_model

#                 st.success("✅ Documents processed successfully!")
#                 st.balloons()
                
# # Main Chat Interface 
# st.markdown("### 💬 Chat with your Medical Documents")



# # Display chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#         st.caption(message["timestamp"])

# # Chat input
# if prompt := st.chat_input("Ask about your medical documents..."):
#     # Add user message to chat history
#     timestamp = time.strftime("%H:%M")
#     st.session_state.messages.append({
#         "role": "user",
#         "content": prompt,
#         "timestamp": timestamp
#     })

#     # Display user message
#     with st.chat_message("user"):
#         st.markdown(prompt)
#         st.caption(timestamp)

#     # Generate response
#     if st.session_state.vectorstore and st.session_state.chat_model:
#         with st.chat_message("assistant"):
#             with st.spinner("🔎 Searching documents..."):
#                 # Retrieve relevant documents
#                 relevant_docs = retrieve_relevant_docs(st.session_state.vectorstore, prompt)

#                 # Create context from relevant documents
#                 context = "\n\n".join([doc.page_content for doc in relevant_docs])

#                 # Create prompt with context
#                 system_prompt = f"""You are MediChat Pro, an intelligent medical document assistant.
#                 Based on the following medical documents, provide accurate and helpful answers.
#                 If the information is not in the documents, clearly state that.
#                 when you are giving an answer make sure that try to take help of llm and give me a full diagnosis of the problem.

#                 Medical Documents:
#                 {context}

#                 User Question: {prompt}

#                 Answer:"""

#                 response = ask_chat_model(st.session_state.chat_model, system_prompt)
                
            
#             st.markdown(response)
#             st.caption(timestamp)












# import streamlit as st  
# from app.ui import pdf_uploader
# from app.pdf_utils import extract_text_from_pdf
# from app.vectorstore_utils import create_faiss_index , retrive_relevent_docs
# from app.chat_utils import get_chat_model, ask_chat_model
# from app.config import GOOGLE_API_KEY
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import time 


# st.set_page_config(
#     page_title="MediChat Pro - Medical Document Assistant",
#     page_icon="🩺",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# st.markdown("""
# <style>
# /* ===== Fixed Top Navbar ===== */
# .navbar {
#     position: fixed;
#     top: 0;
#     left: 0;
#     right: 0;
#     height: 60px;
#     background-color: #009688; /* medical teal */
#     display: flex;
#     align-items: center;
#     justify-content: space-between;
#     padding: 0 1.5rem;
#     color: white;
#     font-family: "Segoe UI", sans-serif;
#     font-size: 1.1rem;
#     font-weight: bold;
#     z-index: 999;
#     box-shadow: 0 2px 6px rgba(0,0,0,0.15);
# }

# /* Left section (icon + title) */
# .navbar-left {
#     display: flex;
#     align-items: center;
# }
# .navbar-left .icon {
#     font-size: 1.5rem;
#     margin-right: 0.6rem;
# }

# /* Right section (profile + logout) */
# .navbar-right {
#     display: flex;
#     align-items: center;
# }
# .navbar-right .profile {
#     width: 34px;
#     height: 34px;
#     border-radius: 50%;
#     background-color: #ffffff33;
#     margin-right: 1rem;
#     display: flex;
#     align-items: center;
#     justify-content: center;
#     font-size: 1.2rem;
# }
# .navbar-right .logout-btn {
#     background-color: #ff4b4b;
#     color: white;
#     padding: 0.4rem 0.9rem;
#     border-radius: 0.4rem;
#     text-decoration: none;
#     font-size: 0.9rem;
#     font-weight: 500;
#     transition: background 0.2s ease-in-out;
# }
# .navbar-right .logout-btn:hover {
#     background-color: #ff3333;
# }

# /* Push content below navbar */
# .main {
#     margin-top: 70px;
# }

# /* ===== Sidebar ===== */
# [data-testid="stSidebar"] {
#     background-color: #ffffff;
#     border-right: 2px solid #e6f0f5;
# }

# /* ===== Chat Messages ===== */
# .chat-message {
#     padding: 1rem;
#     border-radius: 0.6rem;
#     margin-bottom: 1rem;
#     display: flex;
#     flex-direction: column;
#     box-shadow: 0 2px 6px rgba(0,0,0,0.08);
# }
# .chat-message.user {
#     background-color: #2b313e;
#     color: white;
# }
# .chat-message.assistant {
#     background-color: #e8f7f3;
#     color: #00332e;
# }

# /* ===== Buttons ===== */
# .stButton > button {
#     background-color: #009688;
#     color: white;
#     border-radius: 0.5rem;
#     border: none;
#     padding: 0.5rem 1.2rem;
#     font-weight: 600;
#     transition: all 0.2s ease-in-out;
# }
# .stButton > button:hover {
#     background-color: #00796b;
#     transform: scale(1.02);
# }

# /* ===== File Upload Section ===== */
# .upload-section {
#     margin-top: 1rem;
#     padding: 1rem;
#     border: 2px dashed #009688;
#     border-radius: 0.5rem;
#     background-color: #f0faf9;
#     text-align: center;
#     color: #004d40;
#     font-style: italic;
# }
# </style>

# <!-- Navbar HTML -->
# <div class="navbar">
#     <div class="navbar-left">
#         <span class="icon">🩺</span> MediChat Pro - Medical Document Assistant
#     </div>
#     <div class="navbar-right">
#         <div class="profile">👤</div>
#         <a href="#" class="logout-btn">Logout</a>
#     </div>
# </div>

# <div class="main">
# """, unsafe_allow_html=True)

# # Sidebar for document upload
# with st.sidebar:
#     st.markdown("### 📂 Document Upload")
#     st.markdown("Upload your medical documents to start chatting!")

#     uploaded_files = pdf_uploader()

#     if uploaded_files:
#         st.success(f"📄 {len(uploaded_files)} document(s) uploaded")

#         # Process documents
#         if st.button("🚀 Process Documents", type="primary"):
#             with st.spinner("Processing your medical documents..."):
#                 # Extract text from all PDFs
#                 all_texts = []
#                 for file in uploaded_files:
#                     text = extract_text_from_pdf(file)
#                     all_texts.append(text)

#                 # Split texts into chunks
#                 text_splitter = RecursiveCharacterTextSplitter(
#                     chunk_size=1000,
#                     chunk_overlap=200,
#                     length_function=len,
#                 )

#                 chunks = []
#                 for text in all_texts:
#                     chunks.extend(text_splitter.split_text(text))

#                 # Create FAISS index
#                 vectorstore = create_faiss_index(chunks)
#                 st.session_state.vectorstore = vectorstore

#                 # Initialize chat model
#                 chat_model = get_chat_model(GOOGLE_API_KEY)
#                 st.session_state.chat_model = chat_model

#                 st.success("✅ Documents processed successfully!")
#                 st.balloons()
                
                
# #Main Chat Interface 
# st.markdown("###  Chat with your Medical Documents")



