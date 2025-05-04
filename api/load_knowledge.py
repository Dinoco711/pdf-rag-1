#!/usr/bin/env python

import os
import asyncio
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Import the RAG pipeline
from rag_pipeline import get_rag_pipeline

# Load environment variables
load_dotenv()

# Set Google API Key
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

async def load_knowledge(knowledge_dir: str, persist_dir: Optional[str] = None) -> None:
    """
    Load knowledge from text files into the RAG pipeline.
    
    Args:
        knowledge_dir: Directory containing knowledge text files
        persist_dir: Optional directory to persist the ChromaDB database
    """
    # Get the RAG pipeline instance
    rag_pipeline = await get_rag_pipeline(persist_directory=persist_dir)
    
    # Get all text files in the knowledge directory
    knowledge_path = Path(knowledge_dir)
    if not knowledge_path.exists() or not knowledge_path.is_dir():
        raise ValueError(f"Knowledge directory not found: {knowledge_dir}")
    
    text_files = list(knowledge_path.glob("*.txt"))
    if not text_files:
        print(f"No text files found in {knowledge_dir}")
        return
    
    print(f"Found {len(text_files)} knowledge files to load")
    
    # Process each text file
    documents = []
    ids = []
    metadata_list = []
    
    for i, file_path in enumerate(text_files):
        try:
            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            if not content:
                print(f"Skipping empty file: {file_path}")
                continue
                
            # Create a document ID and metadata
            doc_id = f"doc_{i}_{file_path.stem}"
            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "title": file_path.stem.replace('_', ' ').title()
            }
            
            # Add the document to our lists
            documents.append(content)
            ids.append(doc_id)
            metadata_list.append(metadata)
            
            print(f"Processed: {file_path.name}")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    if not documents:
        print("No valid documents found to add to the knowledge base")
        return
    
    # Add documents to the RAG pipeline
    try:
        await rag_pipeline.add_documents(
            documents=documents,
            ids=ids,
            metadatas=metadata_list
        )
        print(f"Successfully added {len(documents)} documents to the knowledge base")
    except Exception as e:
        print(f"Error adding documents to the knowledge base: {str(e)}")

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load knowledge from text files into the RAG pipeline")
    parser.add_argument("--knowledge-dir", "-k", required=True, help="Directory containing knowledge text files")
    parser.add_argument("--persist-dir", "-p", help="Optional directory to persist the ChromaDB database")
    args = parser.parse_args()
    
    # Use persist directory from environment variable if not provided
    persist_dir = args.persist_dir or os.environ.get('CHROMA_PERSIST_DIR')
    
    # Load knowledge
    await load_knowledge(args.knowledge_dir, persist_dir)

if __name__ == "__main__":
    asyncio.run(main()) 