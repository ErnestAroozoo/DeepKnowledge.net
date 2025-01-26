# DeepKnowledge.net

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)

An intelligent Q&A system powered by Retrieval-Augmented Generation (RAG).

![Demo Screenshot](https://github.com/ErnestAroozoo/DeepKnowledge.net/blob/main/demo.png)

## Project Overview

DeepKnowledge.net is an advanced chatbot that integrates large language models with your private data sources using Retrieval-Augmented Generation (RAG). This approach provides precise, source-grounded answers while ensuring data privacy.

## Key Features

- **Multi-source Integration**: Seamlessly process content from websites and documents (PDF/DOCX).
- **Source Citation**: Offers transparent references to original data sources for every response.
- **Relevance Scoring**: Efficiently ranks information based on query relevance.
- **Conversational Memory**: Supports context-aware follow-up questions to maintain dialogue continuity.

## Technical Specifications

- **Language Models**: Utilizes DeepSeek-V3 for chat interactions and OpenAI's text-embedding-ada-002 for embeddings.
- **RAG Framework**: Powered by LlamaIndex.
- **Vector Store**: Employs LlamaIndex In-Memory Vector Store for efficient data retrieval.
- **User Interface**: Built with Streamlit for a seamless web experience.

## Installation Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/ErnestAroozoo/DeepKnowledge.net.git
   cd DeepKnowledge.net
   ```

2. Install necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   ```

## Configuration

Update the `.env` file with your API credentials:

```ini
# OpenAI Embeddings Configuration
OPENAI_API_KEY=your-openai-key
OPENAI_API_HOST=https://api.openai.com/v1

# DeepSeek Chat Configuration
DEEPSEEK_API_KEY=your-deepseek-key
DEEPSEEK_API_HOST=https://api.deepseek.com/v1
```

> **Note**: API keys can be obtained from:
> - OpenAI: [OpenAI Platform](https://platform.openai.com)
> - DeepSeek: [DeepSeek Platform](https://platform.deepseek.com)

## Usage Guide

1. Launch the application:
   ```bash
   streamlit run app.py
   ```

2. Add data sources:
   - **Websites**: Input valid URLs for content parsing.
   - **Documents**: Upload PDF/DOCX files for text extraction.

3. Engage with the chatbot by:
   - Asking natural language queries.
   - Following up with questions using chat history.
   - Requesting source verification for responses.

## Supported Data Sources

| Type        | Formats               | Processing Method       |
|-------------|-----------------------|-------------------------|
| Web Content | URLs                  | Web page parsing        |
| Documents   | PDF, DOCX             | Text extraction         |