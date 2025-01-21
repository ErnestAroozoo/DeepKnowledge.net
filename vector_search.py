from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.readers.web import SimpleWebPageReader
from dotenv import load_dotenv
import os
import json

# Load .env
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_HOST')

# Vector Embedding Parameters
SIMILARITY_TOP_K = 5
SIMILARITY_CUTOFF = 0.7
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002",
    api_key=OPENAI_API_KEY,
    api_base=OPENAI_API_BASE
)

# Chat Response Parameters
llm = OpenAI(
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
    api_base=OPENAI_API_BASE
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
    output = [
        {
            'score': doc.score,
            'url': doc.node.id_,
            'text': doc.node.text
        }
        for doc in documents
    ]

    return output

def chat_response(user_question, chat_memory, index):
    """
    Generates an LLM Q&A response based on vector embeddings and conversation memory.
    """
    # Find top-k results
    results = query_vector_store(index, user_question)

    # Case 1: There are relevant results in vector store
    if results:
        json_internal_sources = json.dumps(results, indent=2) # Convert Python dict to JSON string
    # Case 2: No relevant results in vector store
    else:
        json_internal_sources = "No relevant information found using internal data sources."

    # Construct system prompt
    system_prompt = f"""
    You are a knowledgeable assistant specializing in Q&A. Your role is to provide accurate and precise answers based solely on the information from the provided websites. Do not fabricate information, and only answer questions with evidence from the websites.

    Contextual Sources:
    {json_internal_sources}

    Instructions:
    - Prioritize accuracy, relevance, and clarity in your responses.
    - If the provided documents do not contain sufficient information to answer a question, clearly state that the information is insufficient and suggest consulting additional resources.
    - Ensure that all responses adhere strictly to the context and details of the sourced documents.
    - If multiple sources provide conflicting information, highlight these discrepancies and suggest further review.

    Please proceed with your response based on the documents provided.
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
    urls = ["https://finalfantasy.fandom.com/wiki/Minfilia_Warde", "https://finalfantasy.fandom.com/wiki/Warrior_of_Light_(Final_Fantasy_XIV)"]
    documents = load_web_data(urls)

    # Create vector store index
    index = create_vector_store(documents)

    # Start a chat session
    starting_message = "Hello, I can answer any questions you have based on the websites you have provided and will cite the url in our conversation."
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
