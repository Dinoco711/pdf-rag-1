#!/usr/bin/env python3
"""
Knowledge Base Loading Utility

This script provides a command-line interface for loading knowledge data
from text files into the knowledge base.
"""

import os
import sys
import argparse
import asyncio
from dotenv import load_dotenv

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

from knowledge_base import load_knowledge_from_file, get_knowledge_data, add_knowledge_item
from rag_pipeline import get_rag_pipeline

async def load_and_index(filepath):
    """Load knowledge from a file and index it in the RAG pipeline."""
    print(f"Loading knowledge from {filepath}...")
    items = load_knowledge_from_file(filepath)
    print(f"Loaded {len(items)} items from file.")
    
    print("Initializing RAG pipeline...")
    pipeline = await get_rag_pipeline(force_new=True)
    print("Knowledge base updated and indexed successfully.")
    
    # Print total knowledge items
    total_items = len(get_knowledge_data())
    print(f"Total knowledge items in database: {total_items}")

def main():
    parser = argparse.ArgumentParser(description="Load knowledge data from text files")
    parser.add_argument("--file", "-f", help="Path to knowledge text file (one item per line)")
    parser.add_argument("--item", "-i", help="Add a single knowledge item")
    parser.add_argument("--list", "-l", action="store_true", help="List all knowledge items")
    
    args = parser.parse_args()
    
    if args.list:
        items = get_knowledge_data()
        print(f"Knowledge Base ({len(items)} items):")
        for i, item in enumerate(items, 1):
            print(f"{i}. {item}")
        return
    
    if args.item:
        add_knowledge_item(args.item)
        print("Item added to knowledge base.")
        asyncio.run(load_and_index(None))
        return
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} does not exist.")
            return
        asyncio.run(load_and_index(args.file))
        return
    
    parser.print_help()

if __name__ == "__main__":
    main() 