#!/usr/bin/env python

import os
import asyncio
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

# Define documents to add
DOCUMENTS = [
    # Company information
    "Nexobotics is an AI-powered customer service platform that helps businesses transform their customer interactions. Key features include omnichannel support, AI-powered automation, personalized customer experiences, and real-time analytics.",
    "Nexobotics was founded in 2020 with the mission of revolutionizing how businesses interact with their customers using artificial intelligence and machine learning technologies.",
    "Nexobotics offers a subscription-based pricing model with three tiers: Starter, Professional, and Enterprise. Each tier provides different levels of features and support based on business size and needs.",
    
    # Customer service best practices
    "Customer service best practices include developing a customer-centric culture, implementing omnichannel support, personalizing customer interactions, leveraging AI strategically, and gathering and acting on customer feedback.",
    "Active listening is a critical skill for customer service. It involves fully concentrating on what the customer is saying, understanding their needs, and responding appropriately.",
    "Empathy in customer service means putting yourself in the customer's position and understanding their feelings and frustrations. This helps build rapport and trust with customers.",
    "First contact resolution (FCR) is a key metric in customer service that measures the percentage of customer issues resolved in the first interaction, without requiring follow-up contacts.",
    "Customer journey mapping is the process of visualizing the entire customer experience with your brand from their perspective, helping identify pain points and improvement opportunities.",
    
    # AI in customer service
    "AI chatbots can handle routine customer inquiries efficiently, allowing human agents to focus on complex issues.",
    "Natural Language Processing (NLP) enables AI systems to understand, interpret, and respond to human language in a way that is both meaningful and helpful.",
    "Sentiment analysis in customer service uses AI to detect emotions in customer communications, helping businesses respond appropriately to customer feelings.",
    "AI-powered personalization in customer service involves using customer data to provide tailored experiences and recommendations that meet individual customer needs.",
    "Predictive analytics in customer service uses historical data and AI algorithms to anticipate customer needs and potential issues before they arise.",
    
    # Industry-specific insights
    "In the retail industry, AI-powered customer service can provide personalized product recommendations, automate returns processing, and offer 24/7 shopping assistance.",
    "For financial services, AI customer service solutions can enhance security through voice recognition, streamline account inquiries, and provide personalized financial advice.",
    "In healthcare, AI customer service can assist with appointment scheduling, medication reminders, and answering basic health questions while maintaining patient privacy.",
    "The travel and hospitality industry benefits from AI customer service through automated booking systems, personalized travel recommendations, and real-time travel updates."
]

def main():
    """Add documents to the ChromaDB collection directly"""
    try:
        # Create ChromaDB client
        print("Initializing ChromaDB client...")
        chroma_client = chromadb.Client()
        
        # Get or create collection
        collection_name = "knowledge_base_updated"
        try:
            collection = chroma_client.get_collection(name=collection_name)
            print(f"Using existing collection: {collection_name}")
        except Exception:
            collection = chroma_client.create_collection(name=collection_name)
            print(f"Created new collection: {collection_name}")
        
        # Generate document IDs
        doc_ids = [f"direct_doc_{i+1}" for i in range(len(DOCUMENTS))]
        metadata = [{"source": "direct_add", "type": "customer_service_info"} for _ in range(len(DOCUMENTS))]
        
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
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 