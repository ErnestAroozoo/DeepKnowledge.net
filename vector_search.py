from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.deepseek import DeepSeek
from llama_index.core import Settings
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.schema import NodeRelationship
from dotenv import load_dotenv
import os
import json

# Load .env
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_HOST')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_API_BASE = os.getenv('DEEPSEEK_API_HOST')

# Vector Embedding Parameters
SIMILARITY_TOP_K = 5
SIMILARITY_CUTOFF = 0.75
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002",
    api_key=OPENAI_API_KEY,
    api_base=OPENAI_API_BASE
)

# Chat Response Parameters
llm = DeepSeek(
    model="deepseek-chat",
    temperature=0.5,
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_API_BASE
)

def load_document_data(documents_directory):
    """
    Read documents from the ./data subdirectory
    """
    documents = SimpleDirectoryReader(documents_directory).load_data()
    return documents

def load_web_data(urls):
    """
    Read html texts from a list of web urls
    """
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
    return documents

def create_vector_store(documents):
    """
    Create in-memory vector store
    """
    index = VectorStoreIndex.from_documents(documents)
    return index

def query_vector_store(index, query):
    """
    Perform similarity search in vector store to find and retrieve top-k relevant text chunks
    """
    # Configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=SIMILARITY_TOP_K,
    )

    # Configure query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)],
    )

    # Retrieve relevant documents
    documents = query_engine.retrieve(query)

    # Format the output into a dictionary
    output = []
    for doc in documents:
        # Extract source URL from node relationships
        source_relation = doc.node.relationships.get(NodeRelationship.SOURCE)
        source = source_relation.node_id if source_relation else "N/A"
        
        output.append({
            'score': doc.score,
            'source': source,
            'text': doc.node.text
        })

    return output

def chat_response(user_question, chat_memory, index):
    """
    Generates an LLM Q&A response based on vector embeddings and conversation memory.
    """
    # Find top-k results
    results = query_vector_store(index, user_question)

    # Case 1: There are relevant results in vector store
    if results:
        json_internal_sources = json.dumps(results, indent=4) # Convert Python dict to JSON string
    # Case 2: No relevant results in vector store
    else:
        json_internal_sources = "No relevant information found using internal data sources."

    # Construct system prompt
    system_prompt = f"""
    You are a knowledgeable assistant specializing in Q&A. Your primary role is to provide accurate and precise answers strictly based on the information from the provided documents or websites. Do not fabricate information or make assumptions. Respond only with evidence from the Knowledge Base.

    Knowledge Base (JSON):
    {json_internal_sources}

    Instructions:
    - Provide answers that are accurate, relevant, and easy to understand.
    - Do not speculate or provide information beyond what is included in the Knowledge Base.
    - If the provided documents or websites do not contain sufficient information to answer a question, clearly state this and recommend consulting additional resources.
    - Highlight any conflicting information found across sources and suggest further review if necessary.
    - Ensure all responses are formatted in markdown for compatibility.
    - Ensure to properly cite all the relevant sources used in generating your responses.

    Please proceed with your response based solely on the provided documents or websites from the Knowledge Base.
    """
    messages = [ChatMessage(role="system", content=system_prompt)]

    # Extend messages list to include existing chat history
    messages.extend(chat_memory)

    # Append user question to end of chat history
    messages.append(ChatMessage(role="user", content=user_question))

    # Generate response based on current messages
    response = llm.chat(messages)

    # Return LLM response and top-k results
    return response.message.content, results


if __name__ == "__main__":
    # List of web urls
    urls = [
                "https://www.apple.com/newsroom/2024/02/apple-reports-first-quarter-results/",
                "https://www.apple.com/newsroom/2024/05/apple-reports-second-quarter-results/",
                "https://www.apple.com/newsroom/2024/08/apple-reports-third-quarter-results/", 
                "https://www.apple.com/newsroom/2024/10/apple-reports-fourth-quarter-results/"
            ]
    
    # Load urls
    documents = load_web_data(urls)

    # Create vector store index
    index = create_vector_store(documents)

    # Start a chat session
    starting_message = "Hi! I'm your AI assistant, ready to help answer your questions using the resources you've added to the knowledge base. Ask me anything, and I'll provide accurate, relevant answers based on the information available!"
    chat_memory = [ChatMessage(role="assistant", content=starting_message)]
    while True:
        user_question = input("You: ")
        
        # If the user types 'quit', end the chat session
        if user_question.lower() == 'quit':
            break
        
        # Generate a response
        response = chat_response(user_question, chat_memory, index)[0]

        # Print the chatbot's response
        print(f"Bot: {response}")

        # Append to chat memory
        chat_memory.append(ChatMessage(role="user", content=user_question))
        chat_memory.append(ChatMessage(role="assistant", content=response))
