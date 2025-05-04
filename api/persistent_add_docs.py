#!/usr/bin/env python

import os
import asyncio
import argparse
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb

# Load environment variables
load_dotenv()

# Set Google API Key
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Define documents to add - TOS example
TOS_DOCUMENTS = [
    # Terms of Service documents here...
    """NexoChat is an advanced Al chatbot designed specifically for Nexobotics' branding. It seamlessly operates across multiple platforms, 00 social media DMs, WhatsApp chats, and website bubble widgets.""",
    # ... other documents
]

# Define documents for IP Policy
IP_DOCUMENTS = [
    """Nexobotics retains all intellectual property rights to its AI models and solutions.""",
    """Clients receive a license to use the AI but do not own the underlying technology.""",
    """Customizations made for specific clients remain the property of Nexobotics unless explicitly agreed otherwise."""
]

# Define documents for FAQ
FAQ_DOCUMENTS = [
    """Q: What is NexoChat? A: NexoChat is an advanced AI chatbot that handles customer service across multiple platforms.""",
    """Q: How is NexoVoice different? A: NexoVoice is specifically designed for voice interactions over phone calls.""",
    """Q: Can I customize the AI? A: Yes, we offer various customization options to match your brand voice and needs."""
]

# Map of collection names to their documents
COLLECTION_DOCUMENTS = {
    "terms_of_service": TOS_DOCUMENTS,
    "intellectual_property_policy": IP_DOCUMENTS,
    "faq": FAQ_DOCUMENTS
}
    
def main():
    """Add documents to a persistent ChromaDB collection"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Add documents to ChromaDB collection')
    parser.add_argument('--collection', '-c', type=str, default="terms_of_service",
                        choices=COLLECTION_DOCUMENTS.keys(),
                        help='Collection name to create/update')
    args = parser.parse_args()
    
    collection_name = args.collection
    DOCUMENTS = COLLECTION_DOCUMENTS[collection_name]
    
    try:
        # Create persistent directory if it doesn't exist
        persistent_dir = "./chromadb_data"
        os.makedirs(persistent_dir, exist_ok=True)
        
        # Create ChromaDB client
        print(f"Initializing persistent ChromaDB client in {persistent_dir}...")
        chroma_client = chromadb.PersistentClient(path=persistent_dir)
        
        # Get or create collection
        try:
            collection = chroma_client.get_collection(name=collection_name)
            print(f"Using existing collection: {collection_name}")
            
            # Delete existing collection to start fresh
            print(f"Deleting existing collection to start fresh...")
            chroma_client.delete_collection(name=collection_name)
            collection = chroma_client.create_collection(name=collection_name)
            print(f"Created new collection: {collection_name}")
        except Exception:
            collection = chroma_client.create_collection(name=collection_name)
            print(f"Created new collection: {collection_name}")
        
        # Generate document IDs
        doc_ids = [f"{collection_name}_{i+1}" for i in range(len(DOCUMENTS))]
        metadata = [{"source": collection_name, "category": "knowledge_base"} for _ in range(len(DOCUMENTS))]
        
        # Generate embeddings
        print("Generating embeddings for documents...")
        embeddings = []
        embedding_model = "models/embedding-001"
        
        for doc in DOCUMENTS:
            try:
                embedding_response = genai.embed_content(
                    model=embedding_model,
                    content=doc,
                    task_type="retrieval_document"
                )
                embeddings.append(embedding_response["embedding"])
            except Exception as e:
                print(f"Error generating embedding: {str(e)}")
                return
        
        # Add documents to collection
        print(f"Adding {len(DOCUMENTS)} documents to the collection...")
        collection.add(
            documents=DOCUMENTS,
            embeddings=embeddings,
            ids=doc_ids,
            metadatas=metadata
        )
        
        print("Documents added successfully!")
        
        # Test a query to verify
        test_query = "What are the best practices for customer service?"
        print(f"\nTesting query: {test_query}")
        
        # Generate query embedding
        query_embedding_response = genai.embed_content(
            model=embedding_model,
            content=test_query,
            task_type="retrieval_query"
        )
        query_embedding = query_embedding_response["embedding"]
        
        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        # Print retrieved documents
        retrieved_documents = results["documents"][0]
        print("\nRetrieved documents:")
        for i, doc in enumerate(retrieved_documents):
            print(f"{i+1}. {doc}")
        
        print("\nOur persistent database is now ready for the RAG pipeline to use!")
        print(f"Location: {os.path.abspath(persistent_dir)}")
        print(f"Collection name: {collection_name}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 