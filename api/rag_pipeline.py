"""
RAG Pipeline Module - Compatible with google-generativeai 0.8.4
This module provides a Retrieval-Augmented Generation pipeline 
using Google's Generative AI models and ChromaDB for vector storage.
"""

import os
import time
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions

class RAGPipeline:
    """Retrieval-Augmented Generation Pipeline using Google's Generative AI and ChromaDB"""
    
    def __init__(self, 
                 api_key: str,
                 collection_name: str = "knowledge_base",
                 embedding_model: str = "models/embedding-001",
                 generation_model: str = "models/gemini-1.5-flash",
                 persist_directory: Optional[str] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            api_key: Google API key
            collection_name: Name of the ChromaDB collection
            embedding_model: Model to use for embeddings
            generation_model: Model to use for text generation
            persist_directory: Directory to persist the ChromaDB database
        """
        self.api_key = api_key
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.persist_directory = persist_directory
        
        # Configure Google Generative AI
        genai.configure(api_key=api_key)
        
        # Initialize ChromaDB
        self._init_chroma()
        
    def _init_chroma(self):
        """Initialize ChromaDB client and collection"""
        if self.persist_directory:
            self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        else:
            self.chroma_client = chromadb.Client()
        
        # Create or get the collection
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            # Collection doesn't exist, create a new one
            self.collection = self.chroma_client.create_collection(name=self.collection_name)
            print(f"Initialized collection with knowledge base data...")
            
            # Initialize with some basic documents if needed
            self._init_with_sample_data()
            
    def _init_with_sample_data(self):
        """Initialize the collection with sample data if needed"""
        # This is a placeholder for any initial data you want to add
        # For testing purposes, you might want to add some documents
        sample_docs = [
            "Nexobotics helps businesses improve customer service with AI.",
            "Customer satisfaction is critical for business success.",
            "AI chatbots can handle routine customer inquiries efficiently."
        ]
        
        sample_ids = [f"sample_{i}" for i in range(len(sample_docs))]
        sample_metadata = [{"source": "initial_data"} for _ in range(len(sample_docs))]
        
        try:
            # Generate embeddings for documents
            document_embeddings = []
            for doc in sample_docs:
                embedding_response = genai.embed_content(
                    model=self.embedding_model,
                    content=doc,
                    task_type="retrieval_document"
                )
                document_embeddings.append(embedding_response["embedding"])
            
            # Add documents to the collection
            self.collection.add(
                documents=sample_docs,
                embeddings=document_embeddings,
                ids=sample_ids,
                metadatas=sample_metadata
            )
            print(f"Successfully indexed {len(sample_docs)} documents")
        except Exception as e:
            print(f"Error initializing with sample data: {str(e)}")

    async def add_documents(self, documents: List[str], ids: List[str], 
                          metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of text documents to add
            ids: Unique IDs for each document
            metadatas: Optional metadata for each document
        """
        try:
            if len(documents) != len(ids):
                raise ValueError("Number of documents and ids must match")
                
            if metadatas and len(metadatas) != len(documents):
                raise ValueError("Number of metadata items must match number of documents")
            
            # Generate embeddings for documents
            document_embeddings = []
            for doc in documents:
                embedding_response = genai.embed_content(
                    model=self.embedding_model,
                    content=doc,
                    task_type="retrieval_document"
                )
                document_embeddings.append(embedding_response["embedding"])
                
            # Add documents to the collection
            self.collection.add(
                documents=documents,
                embeddings=document_embeddings,
                ids=ids,
                metadatas=metadatas
            )
            
            print(f"Successfully added {len(documents)} documents to the collection")
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            raise
            
    async def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query the RAG pipeline to get a response based on retrieved context.
        
        Args:
            query_text: The query text
            top_k: Number of top results to retrieve
            
        Returns:
            Dictionary containing the response and retrieved documents
        """
        try:
            # Generate embedding for the query using "retrieval_query" task type
            query_embedding_response = genai.embed_content(
                model=self.embedding_model,
                content=query_text,
                task_type="retrieval_query"
            )
            query_embedding = query_embedding_response["embedding"]
            
            # Query ChromaDB for similar documents - increased to get more context
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Extract the retrieved documents
            retrieved_documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            # Check if any documents were retrieved
            if not retrieved_documents or len(retrieved_documents) == 0:
                return {
                    "response": "That's an interesting question. While I don't have all the specifics at hand, I'd be happy to discuss what we do know about this topic or help with a related question instead.",
                    "documents": [],
                    "distances": []
                }
            
            # Format the prompt for customer service
            query_oneline = query_text.replace("\n", " ")
            prompt = f"""You are MIA, an AI assistant for Pratham. Answer the user's question with confidence and authority.

PRESENTING INFORMATION:
- NEVER mention that you're retrieving or extracting information from anywhere
- Present ALL information as if it's your own direct knowledge, not from passages or a knowledge base
- Speak with the natural confidence of someone who personally knows the answer
- NEVER use phrases like "based on the information," "according to the passages," or "from what I can see"
- Act as if you are the source of the information, not as if you're searching or retrieving it
- You are NOT extracting or searching for information - you simply know it

HANDLING KNOWLEDGE GAPS:
- Only acknowledge a limitation if directly asked about a very specific fact you can't address
- Instead of saying "I don't have that information," pivot naturally to related information
- Frame responses positively around what you know rather than what you don't know
- If you can't answer a specific question, suggest a related insight or ask a follow-up question
- Never apologize for not having information or mention limitations of your knowledge
- Don't say "I don't know" - instead, shift the conversation naturally to related topics

FORMATTING GUIDELINES:
- Use **bold text** for key points, important information, and main answers
- Use *italics* for emphasis or secondary information
- Use bullet points for lists of information
- Format code or technical terms with `backticks`
- Organize complex information in a structured way with headers

RESPONSE STYLE:
- Provide medium-length answers (not too short, not too long)
- Be direct and get straight to the point without small talk
- Do not use greetings at the beginning of your responses
- Maintain a professional but conversational tone
- Only answer what was asked - don't add unnecessary information
- Ensure your responses are informative but easy to read
- Respond in a human-like manner that feels natural and conversational

IMPORTANT RULES:
- NEVER add greetings (like "Hello" or "Hi there") at the beginning of your responses
- Don't refer to users as "customers"
- Don't apologize unnecessarily
- Don't share confidential information unless specifically requested
- Format all responses to be easily readable with appropriate use of markdown
- Keep answers succinct but complete - aim for 3-5 medium-length paragraphs maximum
- NEVER say phrases like "based on the passages" or "according to the information provided"
- NEVER mention that your knowledge is limited or that you don't have certain information
- NEVER reveal that you're retrieving or accessing information from any source

QUESTION: {query_oneline}
"""
            
            # Add the retrieved documents to the prompt
            prompt += "\n\nREFERENCE INFORMATION (DO NOT MENTION THESE AS PASSAGES OR SOURCES):\n"
            for i, passage in enumerate(retrieved_documents):
                passage_oneline = passage.replace("\n", " ")
                prompt += f"INFORMATION {i+1}: {passage_oneline}\n"
            
            print(f"Using model {self.generation_model} for customer service RAG response")
            
            try:
                # Generate the response with tuned parameters for customer service
                model = genai.GenerativeModel(model_name=self.generation_model)
                generation_config = {
                    "temperature": 0.4,     # Lower temperature for more factual responses
                    "top_p": 0.85,          # More focused on high probability tokens
                    "top_k": 40,            
                    "max_output_tokens": 1024,
                }
                response = model.generate_content(prompt, generation_config=generation_config)
                ai_response = response.text
            except Exception as e:
                print(f"Error in generate_content: {str(e)}")
                # Friendly error message
                ai_response = "I'm not quite able to address that right now. Let's try a different approach or question."
            
            return {
                "response": ai_response,
                "documents": retrieved_documents,
                "distances": distances
            }
            
        except Exception as e:
            print(f"Error querying RAG pipeline: {str(e)}")
            return {
                "response": "Let's shift our conversation a bit. Is there something else you'd like to discuss or learn about?",
                "documents": [],
                "distances": []
            }

class MultiCollectionRAGPipeline:
    """RAG Pipeline that can query across multiple ChromaDB collections"""
    
    def __init__(self, 
                 api_key: str,
                 collection_names: List[str],
                 embedding_model: str = "models/embedding-001",
                 generation_model: str = "models/gemini-1.5-flash",
                 persist_directory: Optional[str] = None):
        """
        Initialize the Multi-Collection RAG pipeline.
        
        Args:
            api_key: Google API key
            collection_names: List of ChromaDB collection names to query
            embedding_model: Model to use for embeddings
            generation_model: Model to use for text generation
            persist_directory: Directory to persist the ChromaDB database
        """
        self.api_key = api_key
        self.collection_names = collection_names
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.persist_directory = persist_directory
        
        # Configure Google Generative AI
        genai.configure(api_key=api_key)
        
        # Initialize ChromaDB client
        if self.persist_directory:
            self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        else:
            self.chroma_client = chromadb.Client()
        
        # Initialize collections
        self.collections = {}
        self._init_collections()
    
    def _init_collections(self):
        """Initialize ChromaDB collections"""
        for collection_name in self.collection_names:
            try:
                collection = self.chroma_client.get_collection(name=collection_name)
                print(f"Loaded existing collection: {collection_name}")
                self.collections[collection_name] = collection
            except Exception as e:
                print(f"Could not load collection {collection_name}: {str(e)}")
                # Skip collections that don't exist
                continue
    
    async def query(self, query_text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Query multiple collections and merge results.
        
        Args:
            query_text: The query text
            top_k: Number of top results to retrieve from each collection
            
        Returns:
            Dictionary containing the response and retrieved documents
        """
        try:
            # Generate embedding for the query
            query_embedding_response = genai.embed_content(
                model=self.embedding_model,
                content=query_text,
                task_type="retrieval_query"
            )
            query_embedding = query_embedding_response["embedding"]
            
            # Query each collection and merge results
            all_documents = []
            all_distances = []
            all_collection_names = []
            
            for collection_name, collection in self.collections.items():
                try:
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=top_k,
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    documents = results.get("documents", [[]])[0]
                    distances = results.get("distances", [[]])[0]
                    
                    # Add collection name to each document for tracking
                    for i, doc in enumerate(documents):
                        all_documents.append(doc)
                        all_distances.append(distances[i])
                        all_collection_names.append(collection_name)
                        
                except Exception as e:
                    print(f"Error querying collection {collection_name}: {str(e)}")
                    continue
            
            # Sort all documents by distance
            sorted_results = sorted(zip(all_documents, all_distances, all_collection_names), 
                                   key=lambda x: x[1])
            
            # Take top results across all collections
            top_documents = []
            top_distances = []
            top_collections = [
            "ai_compliance_and_legal_disclaimer",
            "pratham_solanki_personal_details",
        ]
            
            for doc, dist, coll in sorted_results[:top_k*2]:  # Get twice as many for better context
                top_documents.append(doc)
                top_distances.append(dist)
                top_collections.append(coll)
            
            # Check if any documents were retrieved
            if not top_documents:
                return {
                    "response": "That's an interesting question. While I don't have all the specifics at hand, I'd be happy to discuss what we do know about this topic or help with a related question instead.",
                    "documents": [],
                    "distances": [],
                    "collections": []
                }
            
            # Format the prompt for customer service
            query_oneline = query_text.replace("\n", " ")
            prompt = f"""You are MIA, an AI assistant for Pratham. Answer the user's question with confidence and authority.

PRESENTING INFORMATION:
- NEVER mention that you're retrieving or extracting information from anywhere
- Present ALL information as if it's your own direct knowledge, not from passages or a knowledge base
- Speak with the natural confidence of someone who personally knows the answer
- NEVER use phrases like "based on the information," "according to the passages," or "from what I can see"
- Act as if you are the source of the information, not as if you're searching or retrieving it
- You are NOT extracting or searching for information - you simply know it

HANDLING KNOWLEDGE GAPS:
- Only acknowledge a limitation if directly asked about a very specific fact you can't address
- Instead of saying "I don't have that information," pivot naturally to related information
- Frame responses positively around what you know rather than what you don't know
- If you can't answer a specific question, suggest a related insight or ask a follow-up question
- Never apologize for not having information or mention limitations of your knowledge
- Don't say "I don't know" - instead, shift the conversation naturally to related topics

FORMATTING GUIDELINES:
- Use **bold text** for key points, important information, and main answers
- Use *italics* for emphasis or secondary information
- Use bullet points for lists of information
- Format code or technical terms with `backticks`
- Organize complex information in a structured way with headers

RESPONSE STYLE:
- Provide medium-length answers (not too short, not too long)
- Be direct and get straight to the point without small talk
- Do not use greetings at the beginning of your responses
- Maintain a professional but conversational tone
- Only answer what was asked - don't add unnecessary information
- Ensure your responses are informative but easy to read
- Respond in a human-like manner that feels natural and conversational

IMPORTANT RULES:
- NEVER add greetings (like "Hello" or "Hi there") at the beginning of your responses
- Don't refer to users as "customers"
- Don't apologize unnecessarily
- Don't share confidential information unless specifically requested
- Format all responses to be easily readable with appropriate use of markdown
- Keep answers succinct but complete - aim for 3-5 medium-length paragraphs maximum
- NEVER say phrases like "based on the passages" or "according to the information provided"
- NEVER mention that your knowledge is limited or that you don't have certain information
- NEVER reveal that you're retrieving or accessing information from any source

QUESTION: {query_oneline}
"""
            
            # Add the retrieved documents to the prompt
            prompt += "\n\nREFERENCE INFORMATION (DO NOT MENTION THESE AS PASSAGES OR SOURCES):\n"
            for i, (doc, coll) in enumerate(zip(top_documents, top_collections)):
                passage_oneline = doc.replace("\n", " ")
                prompt += f"INFORMATION {i+1}: {passage_oneline}\n"
            
            print(f"Using model {self.generation_model} for customer service RAG response")
            
            try:
                # Generate the response with tuned parameters for customer service
                model = genai.GenerativeModel(model_name=self.generation_model)
                generation_config = {
                    "temperature": 0.4,     # Lower temperature for more factual responses
                    "top_p": 0.85,          # More focused on high probability tokens
                    "top_k": 40,            
                    "max_output_tokens": 1024,
                }
                response = model.generate_content(prompt, generation_config=generation_config)
                ai_response = response.text
            except Exception as e:
                print(f"Error in generate_content: {str(e)}")
                # Friendly error message
                ai_response = "I'm not quite able to address that right now. Let's try a different approach or question."
            
            return {
                "response": ai_response,
                "documents": top_documents,
                "distances": top_distances,
                "collections": top_collections
            }
            
        except Exception as e:
            print(f"Error querying RAG pipeline: {str(e)}")
            return {
                "response": "Let's shift our conversation a bit. Is there something else you'd like to discuss or learn about?",
                "documents": [],
                "distances": [],
                "collections": []
            }

# Singleton instance of the RAG pipeline
_rag_pipeline_instance = None

async def get_rag_pipeline(persist_directory: Optional[str] = None) -> Union[RAGPipeline, MultiCollectionRAGPipeline]:
    """
    Get or create a RAG pipeline instance.
    
    Args:
        persist_directory: Optional directory to persist the ChromaDB database
        
    Returns:
        RAGPipeline instance (either single collection or multi-collection)
    """
    global _rag_pipeline_instance
    
    if _rag_pipeline_instance is None:
        # Get API key from environment
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        # Use the persistent directory from environment or default
        persist_dir = os.environ.get('CHROMA_PERSIST_DIR', "./chromadb_data")
        
        # List of collections to use
        collections = [
            "ai_compliance_and_legal_disclaimer",
            "nexobotics_ai_data_privacy_policy",
            "pratham_solanki_personal_details",
            "ai_refund_cancellation_policy",
            "b2b_partnership_white_label_agreement",
            "nexobotics_ai_usage_policy_guidelines",
            "nexo_robotics_intellectual_property_policy",
            "nexobotics_ai_chatbot_and_voice_agent_terms_of_ser",
            "enterprise_ai_sla_support_agreement"
        ]
        
        # Create Multi-Collection RAG pipeline
        _rag_pipeline_instance = MultiCollectionRAGPipeline(
            api_key=api_key,
            collection_names=collections,
            persist_directory=persist_dir,
            generation_model="models/gemini-1.5-flash",
            embedding_model="models/embedding-001"
        )
    
    return _rag_pipeline_instance

async def query_rag(query_text: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Query the RAG pipeline.
    
    Args:
        query_text: The query text
        top_k: Number of top results to retrieve
        
    Returns:
        Dictionary containing the response and retrieved documents
    """
    rag_pipeline = await get_rag_pipeline()
    return await rag_pipeline.query(query_text, top_k) 