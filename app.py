import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from PIL import Image
import base64
import io
from llama_index.core.llms import ChatMessage, MessageRole
import re
from vector_search import *

# ==========================================================
# Section: Page Config
# ==========================================================
st.set_page_config(
    page_title="DocumentGPT - Your intelligent Q&A AI",
    page_icon="./assets/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# Section: CSS Styling
# ==========================================================
# Layout
st.markdown("""
            <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                .css-15zrgzn {display: none}
                #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}

                /* Remove blank space at top and bottom */ 
                .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    }
                
                /* Remove blank space at the center canvas */ 
                .st-emotion-cache-z5fcl4 {
                    position: relative;
                    top: -62px;
                    }
                
                /* Make the toolbar transparent and the content below it clickable */ 
                .st-emotion-cache-18ni7ap {
                    pointer-events: none;
                    background: rgb(255 255 255 / 0%)
                    }
                .st-emotion-cache-zq5wmm {
                    pointer-events: auto;
                    background: rgb(255 255 255);
                    border-radius: 5px;
                    }
            </style>
                """, unsafe_allow_html=True)

# Footer
st.markdown("""
            <style>
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #232323;
                color: #FFFFFF;
                text-align: center;
                padding: 0px 0;
                font-size: 15px;
                height: 35px;
                line-height: 30px;
            }
            .footer a {
                color: #6464ef;
                text-decoration: none;
            }
            </style>
            <div class="footer">
            <p>Made by <a href='https://github.com/ErnestAroozoo' target='_blank'>Ernest Aroozoo</a> | <a href='https://github.com/ErnestAroozoo/MemeStocks.net' target='_blank'>View on GitHub</a></p>
            </div>
            """, unsafe_allow_html=True)

# Title
def image_to_base64(img_path):
    img = Image.open(img_path)
    img_data = io.BytesIO()
    img.save(img_data, format='PNG')
    img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')
    return img_base64
custom_title = """
                <style>
                    .custom-title {{
                        display: flex;
                        align-items: center;
                        font-family: Arial, sans-serif;
                        color: #FFFFFF;
                    }}
                    .custom-title img {{
                        height: 55px;
                        margin-right: 3px;
                        position: relative;
                        top: -8px;
                    }}
                    .custom-title h1 {{
                        font-size: 2.5rem;
                        margin: 0;
                    }}
                </style>
                <div class="custom-title">
                    <img src="data:image/png;base64,{logo}" alt="Logo">
                    <h2>DocumentGPT</h2>
                </div>
                """
logo_base64 = image_to_base64('./assets/logo.png')
custom_title = custom_title.format(logo=logo_base64)
st.markdown(custom_title, unsafe_allow_html=True)
st.html("<style>[data-testid='stHeaderActionElements'] {display: none;}</style>")

# ==========================================================
# Section: Streamlit UI and Logic
# ==========================================================
def is_valid_url(url):
    """
    Function to validate whether the string is a valid URL
    """
    regex = re.compile(
        r'^(https?://)'  # http:// or https://
        r'(([a-zA-Z0-9_-]+\.)+[a-zA-Z]{2,})'  # domain
        r'(/[a-zA-Z0-9@:%._\+~#=/-]*)*'  # path
        r'(\?[a-zA-Z0-9&=_%-]*)?'  # query string
        r'(#.*)?$'  # fragment locator
    )
    return re.match(regex, url) is not None

# Knowledgebase
st.subheader("Knowledgebase")
with st.container(height=360):
    col1, col2 = st.columns(2)

    # Add Data
    with col1:
        # Add Data
        st.subheader("Add Data")

        # Define default URLs in session state
        if "urls" not in st.session_state:
            st.session_state.urls = [
                "https://finalfantasy.fandom.com/wiki/Minfilia_Warde", 
                "https://finalfantasy.fandom.com/wiki/Warrior_of_Light_(Final_Fantasy_XIV)"
            ]

        # Input for website URL
        website_url = st.text_input("Website URL", "https://finalfantasy.fandom.com/wiki/Minfilia_Warde")

        # Button to add a URL
        if st.button("Add URLs"):
            if is_valid_url(website_url):
                if website_url not in st.session_state.urls:
                    st.session_state.urls.append(website_url)
                    st.success("URL added successfully!")
                else:
                    st.warning("This URL is already in the list.")
            else:
                st.error("Invalid URL. Please enter a valid website link.")

        st.write("") # Empty padding

        # File uploader
        uploaded_file = st.file_uploader("Documents", disabled=True)

    # Vector Store
    with col2:
        st.subheader("Vector Store")
        # Create the DataFrame
        data = {
            "Type": ["Website"] * len(st.session_state.urls),
            "Description": st.session_state.urls,
        }
        vector_store_df = pd.DataFrame(data)

        # Display the DataFrame
        st.dataframe(vector_store_df, hide_index=True)

# Load URLs
if "documents" not in st.session_state:
    st.session_state.documents = load_web_data(st.session_state.urls)

# Create in-memory vector store
if "index" not in st.session_state:
    st.session_state.index = create_vector_store(st.session_state.documents)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Initial assistant message
    initial_message = "Hello! I'm your AI assistant, specialized in answering questions using only the documents you have uploaded to the vector store."
    st.session_state.messages.append(ChatMessage(role="assistant", content=initial_message))

st.subheader("Chat")
with st.container(height=500):
    col1, col2 = st.columns(2)
    # Chatbox
    with col1:
        # Create a placeholder container that holds all the messages
        messages_placeholder = st.container(height=393, border=False)
        
        # Display chat messages from history on app rerun
        with messages_placeholder:
            for message in st.session_state.messages:
                with st.chat_message(message.role):
                    st.markdown(message.content)

        # User Chat Input
        if user_message := st.chat_input("Type your message here..."):
            # Display user message in Chat Display
            with messages_placeholder:
                with st.chat_message("user"):
                    st.markdown(user_message)

            # Generate AI response
            assistant_response, sources = chat_response(user_message, st.session_state.messages, st.session_state.index)

            # Append "user" message to chat history
            st.session_state.messages.append(ChatMessage(role="user", content=user_message))

            # Append "assistant" message to chat history
            st.session_state.messages.append(ChatMessage(role="assistant", content=assistant_response))
            
            # Store sources in session state for display in Sources section
            st.session_state.sources = sources

            # Display assistant message in Chat Display
            with messages_placeholder:
                with st.chat_message("assistant"):
                    st.markdown(assistant_response)

    # Document Sources Table
    with col2:
        with st.container(height=450, border=False):
            st.subheader("Sources")
            # Display the sources if available
            if 'sources' in st.session_state:
                if st.session_state.sources == "":
                    st.markdown("No relevant documents found.")
                else:
                    # Create a DataFrame from the sources dict object
                    sources_df = pd.DataFrame(st.session_state.sources)
                    sources_df = sources_df.rename(columns={
                        'score': 'Relevance',
                        'url': 'URL',
                        'text': 'Text'
                    })
                    # Display DataFrame
                    st.dataframe(sources_df, hide_index=True)
            else:
                st.markdown("Relevant documents will appear here once you start asking questions.")