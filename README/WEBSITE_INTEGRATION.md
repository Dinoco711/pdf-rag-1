# RAG Chatbot Website Integration Guide

This guide explains how to integrate your RAG-powered chatbot into a website for customer service, ensuring it can handle business-specific data effectively.

## Frontend Integration

The `chatbot.js` file provided creates a chat bubble widget that connects to your Flask backend. Here's how it works:

1. It initializes a chat bubble in the bottom-right corner of your website
2. When clicked, it opens a chat window with a welcome message
3. It sends user messages to the `/chat` endpoint and displays responses
4. It supports markdown formatting for rich responses

## Configuring for RAG-Only Mode

Since you prefer to use only the RAG functionality (no switching), here's how to optimize the system:

### Backend Modifications (index.py)

1. **Force RAG mode always**:

```python
# In chat_endpoint() function
# Replace this:
use_rag = request.json.get('use_rag', True)

# With this:
use_rag = True  # Always use RAG regardless of request parameter
```

2. **Remove fallback to non-RAG** (optional):

```python
# In chat_endpoint() function
# You can remove or comment out the entire "if not use_rag or not ai_response:" block
# if you want to strictly use only RAG
```

### Frontend Modifications (chatbot.js)

1. **Simplify API requests** - Remove the use_rag parameter from fetch requests:

```javascript
// In sendMessage() method
body: JSON.stringify({
  message: messageText,
  // No need to include use_rag since backend will always use RAG
}),
```

## Loading Business-Specific Knowledge

To make the chatbot effective for customer service, you need to populate it with relevant business data:

### 1. Organize Your Business Data

Divide your business information into clear, concise documents like:

- Product descriptions and features
- Pricing information
- Return/refund policies
- FAQs
- Troubleshooting guides
- Contact information
- Business hours and locations

### 2. Format Documents for RAG

For each document:

- Keep it focused on a single topic (ideally 2-5 sentences)
- Use clear, straightforward language
- Include relevant keywords that customers might search for

### 3. Add Documents to RAG Database

Edit the `persistent_add_docs.py` file to include your business documents:

```python
# Define documents to add
DOCUMENTS = [
    # Company information
    "Nexobotics is an AI-powered customer service platform that helps businesses transform their customer interactions.",

    # Products and services
    "Our Premium Plan costs $99/month and includes unlimited chat sessions and 24/7 support.",
    "The Enterprise Plan includes custom integrations with your existing CRM and analytics dashboard.",

    # Policies
    "Our refund policy allows customers to request a full refund within 30 days of purchase.",
    "All customer data is encrypted and stored securely in compliance with GDPR regulations.",

    # Common questions
    "To reset your password, click on the 'Forgot Password' link on the login page and follow the instructions sent to your email.",
    "Our platform integrates with popular CRM systems including Salesforce, HubSpot, and Zoho CRM.",

    # Add many more documents covering all aspects of your business
]
```

Then run:

```
python groq-3/api/persistent_add_docs.py
```

### 4. Test Your Business Knowledge

Use the testing tool to verify your business data is being used correctly:

```
python groq-3/direct_api_test.py --message "What is your refund policy?" --verbose
```

## Optimizing for Customer Service

### 1. Adjust RAG Parameters

For customer service, precision is important. Update your RAG settings in `rag_pipeline.py`:

```python
# In the query() method of RAGPipeline class
generation_config = {
    "temperature": 0.3,  # Lower for more factual/consistent responses
    "top_p": 0.85,
    "top_k": 40,
    "max_output_tokens": 1024,
}

# Increase top_k for more comprehensive answers
results = self.collection.query(
    query_embeddings=[query_embedding],
    n_results=5,  # Retrieve more documents for better context
    include=["documents", "metadatas", "distances"]
)
```

### 2. Enhance Prompt Engineering

Update the prompt format in `rag_pipeline.py` to be more customer service oriented:

```python
# In query() method
prompt = f"""You are a helpful customer service assistant. Answer the user's question based on the provided information.
If the information needed isn't in the passages below, politely explain that you don't have that specific information
and suggest how the user could get help (e.g., 'For more details, please contact our support team at support@yourcompany.com').
Be concise, accurate, and friendly.

QUESTION: {query_oneline}
"""
```

### 3. Configure Session Management

To improve user experience, you might want to enable persistent sessions in your frontend:

```javascript
// In the ChatbotWidget constructor
this.sessionId = localStorage.getItem('chatSessionId') || this.generateSessionId();
localStorage.setItem('chatSessionId', this.sessionId);

// In the sendMessage method
body: JSON.stringify({
  message: messageText,
  session_id: this.sessionId  // Include session ID with every request
}),
```

## Deployment Considerations

1. **CORS Settings**: Ensure your server allows requests from your website domain:

   ```python
   # In index.py
   CORS(app, resources={r"/chat": {"origins": "https://yourdomain.com"}})
   ```

2. **API Security**: Consider adding authentication for the chat endpoint in production

3. **Error Handling**: Make sure the frontend gracefully handles server errors:

   ```javascript
   // In the catch block of sendMessage method
   this.addMessage(
     "I'm having trouble connecting to our knowledge base. Please try again in a moment or contact us directly at help@yourcompany.com",
     false
   );
   ```

4. **Performance Monitoring**: Add logging to track common questions and identify knowledge gaps

## Conclusion

By following this guide, you'll have a RAG-powered chatbot that provides accurate, helpful customer service using your business-specific knowledge. Remember to regularly update your knowledge base as your business information changes.

The key to success is comprehensive and well-structured business data in your RAG system. The more quality information you provide, the better your chatbot will perform at assisting customers.
