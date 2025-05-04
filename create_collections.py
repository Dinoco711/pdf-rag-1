#!/usr/bin/env python

import os
import subprocess
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the collections we want to create
COLLECTIONS = [
    "terms_of_service",
    "intellectual_property_policy",
    "faq"
]

def main():
    """Create all ChromaDB collections"""
    parser = argparse.ArgumentParser(description='Create all ChromaDB collections')
    parser.add_argument('--collections', '-c', nargs='+', choices=COLLECTIONS,
                        default=COLLECTIONS, help='Specific collections to create')
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create chromadb_data directory if it doesn't exist
    persistent_dir = os.path.join(script_dir, "chromadb_data")
    os.makedirs(persistent_dir, exist_ok=True)
    
    print("Creating ChromaDB collections:")
    
    for collection in args.collections:
        print(f"\n=== Creating collection: {collection} ===")
        try:
            # Run persistent_add_docs.py with the collection name as an argument
            result = subprocess.run(
                ["python", os.path.join(script_dir, "api", "persistent_add_docs.py"), 
                 "--collection", collection],
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            print(f"Successfully created collection: {collection}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating collection {collection}:")
            print(e.stderr)
    
    print("\nAll collections created successfully!")
    print(f"Collections are stored in: {os.path.abspath(persistent_dir)}")
    print("The chatbot will now use data from all these collections when answering questions.")

if __name__ == "__main__":
    main() 