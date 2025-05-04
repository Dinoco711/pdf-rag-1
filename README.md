[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fvercel%2Fexamples%2Ftree%2Fmain%2Fpython%2Fflask3&demo-title=Flask%203%20%2B%20Vercel&demo-description=Use%20Flask%203%20on%20Vercel%20with%20Serverless%20Functions%20using%20the%20Python%20Runtime.&demo-url=https%3A%2F%2Fflask3-python-template.vercel.app%2F&demo-image=https://assets.vercel.com/image/upload/v1669994156/random/flask.png)

# Nexobotics RAG Chatbot

A customer service chatbot powered by Retrieval-Augmented Generation (RAG) using Google Gemini AI and ChromaDB.

## Project Structure

```
groq-3/
├── api/                      # Backend API
│   ├── index.py              # Main Flask application
│   ├── rag_pipeline.py       # RAG implementation
│   └── persistent_add_docs.py # Knowledge base management
│
├── website/                  # Frontend
│   ├── assets/               # Images and resources
│   └── chatbot.js            # Chat widget implementation
│
├── knowledge/                # Knowledge base content
│   ├── customer_service_best_practices.txt
│   └── nexobotics_info.txt
│
├── direct_api_test.py        # Test script for the API
├── test_chatbot.html         # Test page for the chat widget
├── WEBSITE_INTEGRATION.md    # Guide for website integration
├── USAGE_GUIDE.md            # Guide for using and updating the system
└── requirements.txt          # Python dependencies
```

## Quick Start

1. **Set up the environment**:

   ```
   pip install -r requirements.txt
   ```

2. **Configure your API key**:

   - Edit the `.env` file and add your Google API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

3. **Start the server**:

   ```
   python groq-3/api/index.py
   ```

4. **Test the chatbot**:
   - Open `groq-3/test_chatbot.html` in your browser

## Managing Your Knowledge Base

To update the knowledge that the chatbot can access:

1. Edit `groq-3/api/persistent_add_docs.py` to include your business information
2. Run the script to update the knowledge base:
   ```
   python groq-3/api/persistent_add_docs.py
   ```

## Documentation

The project includes several documentation files:

- `WEBSITE_INTEGRATION.md` - How to integrate the chatbot into your website
- `USAGE_GUIDE.md` - How to use and maintain the system

## Features

- **RAG-Powered Responses**: Answers are based on your knowledge base
- **Persistent Sessions**: Conversation continuity for users
- **Markdown Support**: Rich formatting for responses
- **Customizable UI**: Easy to adapt to your brand

## License

This project is licensed under the MIT License.

# Flask + Vercel

This example shows how to use Flask 3 on Vercel with Serverless Functions using the [Python Runtime](https://vercel.com/docs/concepts/functions/serverless-functions/runtimes/python).

## Demo

https://flask-python-template.vercel.app/

## How it Works

This example uses the Web Server Gateway Interface (WSGI) with Flask to enable handling requests on Vercel with Serverless Functions.

## Running Locally

```bash
npm i -g vercel
vercel dev
```

Your Flask application is now available at `http://localhost:3000`.

## One-Click Deploy

Deploy the example using [Vercel](https://vercel.com?utm_source=github&utm_medium=readme&utm_campaign=vercel-examples):

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fvercel%2Fexamples%2Ftree%2Fmain%2Fpython%2Fflask3&demo-title=Flask%203%20%2B%20Vercel&demo-description=Use%20Flask%203%20on%20Vercel%20with%20Serverless%20Functions%20using%20the%20Python%20Runtime.&demo-url=https%3A%2F%2Fflask3-python-template.vercel.app%2F&demo-image=https://assets.vercel.com/image/upload/v1669994156/random/flask.png)
