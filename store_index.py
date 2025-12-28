from src.helper import load_pdf, text_split, download_hugging_face_embedding
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

extracterd_data = load_pdf('data/')
text_chunks = text_split(extracterd_data)
embeddings_model = download_hugging_face_embedding()


pinecone_client = Pinecone(api_key = PINECONE_API_KEY)

index_name = 'medical-chatbot'

doc_search = PineconeVectorStore.from_texts(
    [t.page_content for t in text_chunks],
    embeddings_model,
    index_name = index_name
)