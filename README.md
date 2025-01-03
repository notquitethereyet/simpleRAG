## Pinecone + Groq RAG System

A simple **Retrieval-Augmented Generation (RAG)** project using **Pinecone** for vector storage and **Groq** for query inference. It processes PDFs, stores embeddings, and answers natural language queries.

### Features

- Process PDFs and store embeddings in Pinecone.
- Search Pinecone for relevant results based on your query.
- Use Groq to infer and refine search queries with context.

### Installation
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirement.txt
```
### Create enviroment Variable in `.env` file
```
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-east-1
GROQ_API_KEY=your-groq-api-key
INDEX_NAME=your-index-name
MODEL_NAME=multilingual-e5-large
```
### Usage
`python main.py`

### Project Layout
```bash
.
├── preprocess.py      # Extract text from PDFs
├── embed.py           # Generate embeddings
├── store.py           # Store embeddings in Pinecone
├── search.py          # Search Pinecone
├── cock.py            # Query inference with Groq
├── config.py          # Environment variables
├── main.py            # The main script
├── .env                   # API keys and config
├── requirements.txt       # Python dependencies
└── README.md              # This file
```
### Why This?

It’s simple, modular, and helps you play with Pinecone and Groq for RAG tasks. Great for learning or building basic RAG systems. I did this solely to mess around with my friend's resume :D 
