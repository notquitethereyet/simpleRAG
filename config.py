from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Pinecone configuration
API_KEY = os.getenv("PINECONE_API_KEY")
ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("INDEX_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
