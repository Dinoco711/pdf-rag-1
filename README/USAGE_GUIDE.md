# Nexobotics RAG Chatbot Usage Guide

## Cleaning the Codebase

Your current implementation has several redundant scripts. Let's consolidate them:

1. Keep only these essential files:

   - `api/index.py` - Main API server
   - `api/rag_pipeline.py` - Core RAG functionality
   - `api/persistent_add_docs.py` - Document management
   - `direct_api_test.py` - Testing tool

2. Remove these redundant files:
   - `api/add_docs.py`
   - `api/direct_add_docs.py`
   - `api/simple_test.py`
   - `api/rag_test.py`
   - `api_test.py`

## How to Use the RAG Chatbot

### 1. Adding/Updating Documents

The easiest way to update your knowledge base is by editing the `persistent_add_docs.py` file:

1. Open `api/persistent_add_docs.py`
2. Modify the `DOCUMENTS` list to add, remove, or update content
3. Run the script to update your database:
   ```
   python groq-3/api/persistent_add_docs.py
   ```

Example of adding new documents:

```python
# Define documents to add
DOCUMENTS = [
    # Your existing documents...

    # Add new documents here
    "Nexobotics offers 24/7 customer support for all enterprise customers.",
    "Our AI-powered chatbots can reduce customer service costs by up to 30%."
]
```

### 2. Customizing the System

#### A. Change RAG Parameters

To modify how the RAG system works, edit `rag_pipeline.py`:

1. Adjust the number of retrieved documents:

   ```python
   # In direct_api_test.py when calling the API
   test_chat_api("Your question?", use_rag=True, top_k=3)  # Change 3 to desired number

   # In rag_pipeline.py
   async def query_rag(query_text: str, top_k: int = 5):  # Change default from 5
   ```

2. Change the model used:

   ```python
   # In get_rag_pipeline() function
   _rag_pipeline_instance = RAGPipeline(
       api_key=api_key,
       collection_name="customer_service_best_practices",
       generation_model="models/gemini-1.5-flash",  # Change to another model
       embedding_model="models/embedding-001"       # For embeddings
   )
   ```

3. Adjust generation parameters:
   ```python
   # In query() method of RAGPipeline class
   generation_config = {
       "temperature": 0.7,  # Higher = more creative, lower = more focused
       "top_p": 0.95,       # Controls diversity
       "top_k": 40,         # Limits vocabulary options
       "max_output_tokens": 1024,  # Maximum response length
   }
   ```

#### B. Change the Chatbot Personality

To modify how the chatbot responds in non-RAG mode:

1. Edit the `CONTEXT` variable in `index.py`:
   ```python
   CONTEXT = """You are NOVA, a proactive and adaptable customer service agent...
   # Modify the instructions to change personality, tone, etc.
   """
   ```

### 3. Basic Usage

#### Start the Server

```
python groq-3/api/index.py
```

#### Test Using the Command Line

```
python groq-3/direct_api_test.py --message "What are the best practices for customer service?"
```

Options:

- `--no-rag` - Disable RAG (use standard chat mode)
- `--verbose` - Show detailed request/response information

#### Integrate with Your Application

Send POST requests to `http://localhost:5000/chat`:

```json
{
  "message": "Your question here",
  "session_id": "unique_user_id",
  "use_rag": true
}
```

The response will contain the chatbot's answer:

```json
{
  "response": "The chatbot's detailed answer..."
}
```

## Performance Tips

1. Using persistent storage significantly improves startup time on restarts
2. Add more relevant documents to improve answer quality
3. Adjust the `temperature` parameter (lower for more factual responses, higher for more creative ones)
4. If responses are too long, adjust `max_output_tokens` in the generation config
