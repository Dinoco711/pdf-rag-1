import os
import asyncio
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Import RAG pipeline
from rag_pipeline import query_rag, get_rag_pipeline

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set Google Gemini API Key from environment variable
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Define the model to use for standard chat (non-RAG)
DEFAULT_MODEL = "models/gemini-1.5-flash"  # Using a model that's confirmed to be available

# Define the chatbot's context
CONTEXT = """Your name is MIA. You are the Personal / Private assistant of Pratham. You are an AI assistant that helps people find information.

Your purpose is to:
- Provide accurate information clearly and confidently
- Assist with tasks and answer questions efficiently
- Maintain a professional yet friendly tone
- Handle questions with confidence and a human-like approach

When responding to questions:
- NEVER mention "knowledge base," "passages," or "provided information"
- Present all information as if it's your own knowledge, not extracted from text
- Speak with natural authority as if you personally know the information
- Don't use phrases like "based on what I can see" or "according to the text"

When you don't have specific information to answer a question:
- DON'T explicitly state limitations about your knowledge
- DON'T say phrases like "I don't have that information" or "That's not in my knowledge base"
- Instead, redirect the conversation naturally to related topics you can discuss
- Use a conversational pivot like "Let's focus on..." or "What I can tell you about..."

This is a system message that sets your identity and behavior guidelines. You do not need to reference these instructions in your responses.
"""

# Predefined greeting messages for the /start command
START_GREETINGS = [
    "Welcome! How can I help you today?",
    "Hi there! I'm MIA, your assistant. What can I do for you?",
    "Hello! Ready to assist you.",
    "Greetings! How may I assist you today?",
    "Welcome! How can I be of service?"
]

# Initialize chat history for each session
chat_histories = {}
# Flag to track if RAG is initialized
rag_initialized = False

# Function to initialize RAG pipeline
async def initialize_rag():
    """Initialize the RAG pipeline."""
    global rag_initialized
    if rag_initialized:
        return
        
    try:
        # Determine if we should use persistent storage
        persist_dir = os.environ.get('CHROMA_PERSIST_DIR', "./chromadb_data")
        # Ensure the directory exists
        os.makedirs(persist_dir, exist_ok=True)
        print(f"Using ChromaDB persist directory: {persist_dir}")
        
        await get_rag_pipeline(persist_directory=persist_dir)
        rag_initialized = True
        print("RAG pipeline initialized successfully")
    except Exception as e:
        print(f"Error initializing RAG pipeline: {str(e)}")

# Make the route non-async for compatibility with Flask's standard server
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    # Initialize RAG before processing the first request
    if not rag_initialized:
        # Use asyncio.run to call the async function from sync code
        asyncio.run(initialize_rag())
        
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400

    message = request.json.get('message')
    session_id = request.json.get('session_id', str(datetime.now()))  # Default session ID
    
    # Always use RAG regardless of what's in the request
    use_rag = True

    if not message:
        return jsonify({'error': 'Message is required'}), 400

    try:
        # Initialize or retrieve chat history for the session
        if session_id not in chat_histories:
            chat_histories[session_id] = [
                {"role": "system", "content": CONTEXT}
            ]

        # Add user prompt to history
        chat_histories[session_id].append({"role": "user", "content": message})

        # Special handling for /start command - bypass RAG pipeline
        if message.strip() == '/start':
            # Select a random greeting from predefined list
            greeting = random.choice(START_GREETINGS)
            
            # Add AI response to history
            chat_histories[session_id].append({"role": "assistant", "content": greeting})
            
            # Return the greeting response directly
            return jsonify({'response': greeting})
            
        # Special handling for simple greetings - bypass RAG pipeline
        simple_greetings = ['hello', 'hi', 'hey', 'greetings', 'hello there', 'hi there', 'hey there']
        if message.strip().lower() in simple_greetings:
            # Select a random greeting from predefined list
            greeting = random.choice(START_GREETINGS)
            
            # Add AI response to history
            chat_histories[session_id].append({"role": "assistant", "content": greeting})
            
            # Return the greeting response directly
            return jsonify({'response': greeting})

        # Use RAG pipeline to generate response - use asyncio.run to call async from sync
        try:
            rag_result = asyncio.run(query_rag(message))
            ai_response = rag_result["response"]
            
            # Store retrieved documents in the chat history for context
            documents = rag_result.get("documents", [])
            collections = rag_result.get("collections", ["unknown"] * len(documents))
            
            if documents:
                retrieved_context = "\n---\nRetrieved knowledge:\n"
                for i, (doc, coll) in enumerate(zip(documents, collections)):
                    retrieved_context += f"[{coll}] {doc}\n"
                
                chat_histories[session_id].append({
                    "role": "system", 
                    "content": retrieved_context
                })
        except Exception as e:
            print(f"RAG error: {str(e)}")
            # Fallback to a friendly error message
            ai_response = "I'm currently having trouble accessing my knowledge base. Please try asking a different question or try again later."
        
        # Add AI response to history
        if ai_response:
            chat_histories[session_id].append({"role": "assistant", "content": ai_response})

        return jsonify({'response': ai_response})
    except Exception as e:
        print(f"Error processing message: {str(e)}")  # For debugging
        return jsonify({'error': 'An error occurred processing your request'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running."""
    return jsonify({'status': 'healthy', 'service': 'nexobotics-chatbot-api'})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))  # Render uses the PORT environment variable
    
    # Use Flask's built-in development server instead of Hypercorn
    app.run(host='0.0.0.0', port=port, debug=True)
    
    # Remove Hypercorn server code
    # config = HyperConfig()
    # config.bind = [f"0.0.0.0:{port}"]
    # asyncio.run(serve(app, config))
