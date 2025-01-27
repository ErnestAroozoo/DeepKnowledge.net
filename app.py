import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from PIL import Image
import base64
import io
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import NodeRelationship
import re
from vector_search import *

# ==========================================================
# Section: Page Config
# ==========================================================
st.set_page_config(
    page_title="DeepKnowledge.net - Your intelligent Q&A AI",
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
            <p>Made by <a href='https://github.com/ErnestAroozoo' target='_blank'>Ernest Aroozoo</a> | <a href='https://github.com/ErnestAroozoo/DeepKnowledge.net' target='_blank'>View on GitHub</a></p>
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
                    <h2>DeepKnowledge.net</h2>
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

def get_urls_from_index(index):
    """
    Helper function to extract unique source URLs from vector store index
    """
    urls = set()
    for node_id in index.docstore.docs.keys():
        node = index.docstore.get_node(node_id)
        source_relation = node.relationships.get(NodeRelationship.SOURCE)
        if source_relation and source_relation.node_id:
            urls.add(source_relation.node_id)
    return sorted(urls)

def clear_url_input():
    """
    Helper function to clear user input
    """
    # Store submitted URL before clearing
    st.session_state.submitted_url = st.session_state.website_input
    # Clear the text input widget
    st.session_state.website_input = ""

# Define default URLs in session state
if "index" not in st.session_state:
    default_urls = [
        "https://www.apple.com/newsroom/2024/02/apple-reports-first-quarter-results/",
        "https://www.apple.com/newsroom/2024/05/apple-reports-second-quarter-results/",
        "https://www.apple.com/newsroom/2024/08/apple-reports-third-quarter-results/", 
        "https://www.apple.com/newsroom/2024/10/apple-reports-fourth-quarter-results/"
    ]
    st.session_state.documents = load_web_data(default_urls)
    st.session_state.index = create_vector_store(st.session_state.documents)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Initial assistant message
    initial_message = "Hi! I'm your AI assistant, ready to help answer your questions using the resources you've added to the knowledge base. Ask me anything, and I'll provide accurate, relevant answers based on the information available!"
    st.session_state.messages.append(ChatMessage(role="assistant", content=initial_message))

# Initialize website url input
    if 'submitted_url' not in st.session_state:
        st.session_state.submitted_url = None
    if 'website_input' not in st.session_state:
        st.session_state.website_input = ""

@st.fragment()
def knowledge_base_layout():
    """
    Fragmented UI component for Knowledge Base
    """
    st.write("") # Empty padding
    with st.expander(":material/database: Knowledge Base", expanded=True):
        col1, col2 = st.columns(2)

        # Add Data
        with col1:
            st.subheader(":material/library_add: Add Data")

            # Add Data (Website URL)
            st.text_input(
                label="Website URL",
                placeholder="Type a website URL and then press enter (e.g. https://website.com)",
                key="website_input",
                on_change=clear_url_input
            )

            # Process submitted URL if exists
            if st.session_state.submitted_url:
                website_url = st.session_state.submitted_url
                # Reset submitted URL immediately to prevent reprocessing
                st.session_state.submitted_url = None

                # Original validation and processing logic
                if is_valid_url(website_url):
                    current_urls = get_urls_from_index(st.session_state.index)
                    if website_url not in current_urls:
                        new_urls = current_urls + [website_url]
                        
                        try:
                            # Attempt to load data with new URLs
                            st.session_state.documents = load_web_data(new_urls)
                            st.session_state.index = create_vector_store(st.session_state.documents)
                            st.success("URL added successfully!", icon=":material/success:")
                        except Exception as e:
                            st.error(f"Error loading website.", icon=":material/error:")
        
                    else:
                        st.warning("This URL is already in the knowledge base.", icon=":material/warning:")
                else:
                    st.error("Invalid URL. Please enter a valid website link.", icon=":material/error:")

            st.write("") # Empty padding

            # Add Data (Document)
            uploaded_file = st.file_uploader("Document", type=["pdf", "docx"], disabled=True)

        # Data Source
        with col2:
            st.subheader(":material/database: Knowledge Base")
            # Get URLs directly from index
            index_urls = get_urls_from_index(st.session_state.index)
            
            # Create DataFrame from index data
            vector_store_df = pd.DataFrame({
                "Type": ["Website"] * len(index_urls),
                "Source": index_urls
            })

            # Display the DataFrame
            st.dataframe(vector_store_df, hide_index=True, use_container_width=True)
knowledge_base_layout()

@st.fragment
def chat_layout():
    """
    Fragmented UI component for Chat
    """
    with st.container(height=515, border=False):
        col1, col2 = st.columns(2)
        # Chatbox
        with col1:
            st.subheader(":material/forum: Chat")
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
            st.subheader(":material/fact_check: Relevant Sources")
            # Display the sources if available
            if 'sources' in st.session_state:
                if len(st.session_state.sources) == 0:
                    st.warning("No relevant documents or websites found in knowledge base.", icon=":material/warning:")
                else:
                    # Create a DataFrame from the sources dict object
                    sources_df = pd.DataFrame(st.session_state.sources)
                    sources_df = sources_df.rename(columns={
                        'score': 'Relevance',
                        'source': 'Source',
                        'text': 'Text'
                    })
                    # Display DataFrame
                    st.dataframe(sources_df, hide_index=True, use_container_width=True)
            else:
                st.info("Relevant documents or websites from the knowledge base will appear here once you start asking questions.", icon=":material/info:")
chat_layout()

# ==========================================================
# Section: CSS Footer
# ==========================================================
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
            <p>Made by <a href='https://github.com/ErnestAroozoo' target='_blank'>Ernest Aroozoo</a> | <a href='https://github.com/ErnestAroozoo/DeepKnowledge.net' target='_blank'>View on GitHub</a></p>
            </div>
            """, unsafe_allow_html=True)
