# PDF Knowledge Base Processing

This system allows you to automatically process PDF files and integrate them into your RAG (Retrieval-Augmented Generation) chatbot. The system will:

1. Extract text from PDFs
2. Use AI to determine appropriate collection names based on content
3. Split the content into manageable chunks
4. Create ChromaDB collections for each PDF
5. Automatically update the RAG pipeline to include these collections

## Setup

1. Make sure you have the required packages installed:

   ```
   pip install PyPDF2 google-generativeai chromadb
   ```

2. Place your PDF files in the `knowledge_base` directory.

## Usage

### Process PDF Files

To process all PDF files in the knowledge base directory:

```bash
python process_pdf_docs.py
```

### Custom Options

You can specify a different directory or file pattern:

```bash
python process_pdf_docs.py --pdf_dir custom_directory --pdf_pattern "*.pdf"
```

## How It Works

1. **PDF Text Extraction**: The script extracts all text from your PDF files.

2. **Intelligent Collection Naming**: Using Google's Gemini model, the system analyzes the content and generates a meaningful collection name.

3. **Text Chunking**: Long documents are split into chunks with some overlap for better processing.

4. **ChromaDB Integration**: Each PDF's content is stored in its own ChromaDB collection.

5. **Automatic RAG Pipeline Update**: The system updates your RAG pipeline to include the new collections.

## Configuration

You can adjust these settings in the `process_pdf_docs.py` file:

- `CHUNK_SIZE`: Size of each text chunk (default: 1000 characters)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200 characters)
- `COLLECTION_NAME_PREFIX`: Prefix for automatic collection names (default: "pdf\_")

## Outputs

- `pdf_collections.json`: Contains a list of all PDF-based collections
- ChromaDB collections in your persistent storage directory

## Troubleshooting

If you encounter any issues:

1. Check that your PDF files are text-based (not scanned images)
2. Ensure your Google API key is properly set in your `.env` file
3. Make sure you have sufficient permissions to create and write files

For more advanced debugging, check the console output for detailed error messages.
