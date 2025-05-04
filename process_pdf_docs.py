#!/usr/bin/env python

import os
import glob
import argparse
import re
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
import PyPDF2

# Load environment variables
load_dotenv()

# Set Google API Key
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Global variables
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
COLLECTION_NAME_PREFIX = "pdf_"  # Prefix for collection names

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks for better processing."""
    if not text:
        return []
        
    # Clean the text - remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
    
    return chunks

def generate_collection_name(pdf_path: str, pdf_content: str) -> str:
    """Generate a collection name using AI based on PDF content."""
    try:
        filename = os.path.basename(pdf_path)
        
        # First try to generate a collection name using AI
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        
        # Truncate content if too long
        content_summary = pdf_content[:5000] + "..." if len(pdf_content) > 5000 else pdf_content
        
        prompt = f"""Based on the following PDF content, generate a concise collection name that describes the document's content.
The name should be lowercase, with words separated by underscores, and be descriptive of the document's main topic.
Do not include the word "pdf" in the name.
For example: "customer_service_best_practices", "product_specifications", etc.

PDF Filename: {filename}
PDF Content: {content_summary}

Collection name (lowercase with underscores, max 5 words):"""
        
        response = model.generate_content(prompt)
        suggested_name = response.text.strip().lower()
        
        # Clean the name - remove any characters that aren't lowercase letters, numbers, or underscores
        suggested_name = re.sub(r'[^a-z0-9_]', '_', suggested_name)
        
        # Ensure it's not too long
        if len(suggested_name) > 50:
            suggested_name = suggested_name[:50]
            
        return suggested_name
    
    except Exception as e:
        print(f"Error generating collection name: {str(e)}")
        # Fallback to using the file name
        name = os.path.splitext(os.path.basename(pdf_path))[0].lower()
        name = re.sub(r'[^a-z0-9_]', '_', name)
        return f"{COLLECTION_NAME_PREFIX}{name}"

def process_pdf(pdf_path: str, chroma_client: chromadb.PersistentClient, persist_directory: str) -> Optional[str]:
    """Process a PDF file and add its content to ChromaDB."""
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return None
    
    # Extract text from PDF
    print(f"Extracting text from {pdf_path}...")
    pdf_content = extract_text_from_pdf(pdf_path)
    if not pdf_content:
        print(f"Could not extract content from {pdf_path}")
        return None
    
    # Generate collection name
    collection_name = generate_collection_name(pdf_path, pdf_content)
    print(f"Generated collection name: {collection_name}")
    
    # Create or get collection
    try:
        try:
            collection = chroma_client.get_collection(name=collection_name)
            print(f"Using existing collection: {collection_name}")
            
            # Delete existing collection to start fresh
            print(f"Deleting existing collection to start fresh...")
            chroma_client.delete_collection(name=collection_name)
        except Exception:
            pass  # Collection doesn't exist yet, which is fine
            
        # Create new collection
        collection = chroma_client.create_collection(name=collection_name)
        print(f"Created new collection: {collection_name}")
        
        # Chunk the text
        chunks = chunk_text(pdf_content)
        if not chunks:
            print(f"No content chunks generated from {pdf_path}")
            return None
        
        # Generate document IDs and metadata
        doc_ids = [f"{collection_name}_{i+1}" for i in range(len(chunks))]
        metadatas = [{
            "source": os.path.basename(pdf_path),
            "category": "pdf_document",
            "chunk": i+1,
            "total_chunks": len(chunks)
        } for i in range(len(chunks))]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = []
        embedding_model = "models/embedding-001"
        
        for chunk in chunks:
            try:
                embedding_response = genai.embed_content(
                    model=embedding_model,
                    content=chunk,
                    task_type="retrieval_document"
                )
                embeddings.append(embedding_response["embedding"])
            except Exception as e:
                print(f"Error generating embedding: {str(e)}")
                return None
        
        # Add documents to collection
        print(f"Adding {len(chunks)} chunks to the collection...")
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=doc_ids,
            metadatas=metadatas
        )
        
        print(f"Successfully processed {pdf_path}")
        return collection_name
        
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        return None

def update_collection_list(collection_names: List[str], collection_list_file: str = "pdf_collections.json"):
    """Update the list of PDF collections in a JSON file."""
    try:
        # Create file if it doesn't exist
        if not os.path.exists(collection_list_file):
            with open(collection_list_file, 'w') as f:
                json.dump({"pdf_collections": []}, f)
        
        # Read current collections
        with open(collection_list_file, 'r') as f:
            data = json.load(f)
        
        # Update collections (avoiding duplicates)
        existing_collections = set(data.get("pdf_collections", []))
        for name in collection_names:
            if name:  # Skip None values
                existing_collections.add(name)
        
        # Write back
        data["pdf_collections"] = list(existing_collections)
        with open(collection_list_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Updated collection list in {collection_list_file}")
        
    except Exception as e:
        print(f"Error updating collection list: {str(e)}")

def update_rag_pipeline_collections(pdf_collections_file: str = "pdf_collections.json", rag_pipeline_file: str = "api/rag_pipeline.py"):
    """Update the RAG pipeline to include the PDF collections."""
    try:
        # Read the current PDF collections
        if not os.path.exists(pdf_collections_file):
            print(f"Collection list file not found: {pdf_collections_file}")
            return
            
        with open(pdf_collections_file, 'r') as f:
            data = json.load(f)
        
        pdf_collections = data.get("pdf_collections", [])
        if not pdf_collections:
            print("No PDF collections to add to RAG pipeline")
            return
        
        # Read the current RAG pipeline code
        if not os.path.exists(rag_pipeline_file):
            print(f"RAG pipeline file not found: {rag_pipeline_file}")
            return
            
        with open(rag_pipeline_file, 'r') as f:
            code = f.read()
        
        # Find the collections list in the code
        pattern = r"collections\s*=\s*\[(.*?)\]"
        match = re.search(pattern, code, re.DOTALL)
        
        if not match:
            print(f"Could not find collections list in {rag_pipeline_file}")
            return
        
        # Parse the current collections
        current_collections_text = match.group(1)
        current_collections = [c.strip(' "\'\n\t,') for c in current_collections_text.split(",")]
        current_collections = [c for c in current_collections if c]  # Remove empty strings
        
        # Add new collections
        all_collections = set(current_collections)
        for collection in pdf_collections:
            all_collections.add(collection)
        
        # Format the new collections list
        new_collections_text = ",\n            ".join([f'"{c}"' for c in all_collections])
        new_collections_block = f"collections = [\n            {new_collections_text}\n        ]"
        
        # Replace in the code
        updated_code = re.sub(pattern, new_collections_block, code, flags=re.DOTALL)
        
        # Write the updated code back
        with open(rag_pipeline_file, 'w') as f:
            f.write(updated_code)
            
        print(f"Updated RAG pipeline file {rag_pipeline_file} with {len(pdf_collections)} PDF collections")
        
    except Exception as e:
        print(f"Error updating RAG pipeline: {str(e)}")

def main():
    """Process PDF files and add them to ChromaDB."""
    parser = argparse.ArgumentParser(description='Process PDF files and add to ChromaDB')
    parser.add_argument('--pdf_dir', type=str, default='knowledge_base',
                        help='Directory containing PDF files')
    parser.add_argument('--pdf_pattern', type=str, default='*.pdf',
                        help='Pattern to match PDF files')
    args = parser.parse_args()
    
    # Get the absolute path of the PDF directory
    pdf_dir = os.path.abspath(args.pdf_dir)
    if not os.path.exists(pdf_dir):
        print(f"PDF directory not found: {pdf_dir}")
        return
    
    # Set up ChromaDB
    persistent_dir = os.environ.get('CHROMA_PERSIST_DIR', "./chromadb_data")
    os.makedirs(persistent_dir, exist_ok=True)
    
    print(f"Initializing ChromaDB client in {persistent_dir}...")
    chroma_client = chromadb.PersistentClient(path=persistent_dir)
    
    # Find PDF files
    pdf_pattern = os.path.join(pdf_dir, args.pdf_pattern)
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        print(f"No PDF files found matching {pdf_pattern}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process each PDF file
    processed_collections = []
    for pdf_file in pdf_files:
        print(f"\nProcessing {pdf_file}...")
        collection_name = process_pdf(pdf_file, chroma_client, persistent_dir)
        if collection_name:
            processed_collections.append(collection_name)
    
    # Update collection list
    if processed_collections:
        update_collection_list(processed_collections)
        update_rag_pipeline_collections()
        
        print("\nAll PDFs processed successfully!")
        print(f"Created {len(processed_collections)} collections: {', '.join(processed_collections)}")
        print("Run your API to use these collections in your RAG pipeline")
    else:
        print("\nNo collections were created. Check the logs for errors.")

if __name__ == "__main__":
    main() 