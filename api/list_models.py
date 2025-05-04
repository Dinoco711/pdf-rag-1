#!/usr/bin/env python

import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Google API Key
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# List available models
def list_models():
    models = genai.list_models()
    print("Available models:")
    for model in models:
        print(f"- {model.name}")
        
    # Print detailed info about embedding models
    print("\nEmbedding models:")
    for model in models:
        if "embedding" in model.name.lower():
            print(f"\nModel: {model.name}")
            print(f"   Supported generation methods: {model.supported_generation_methods}")

if __name__ == "__main__":
    list_models() 