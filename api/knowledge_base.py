"""
Knowledge Base Module

This module contains the knowledge base data as a list of strings.
Each string represents a chunk of knowledge that can be indexed and retrieved by the RAG pipeline.
"""

# Sample knowledge data
KNOWLEDGE_DATA = [
    "Nexobotics specializes in AI-powered customer service automation solutions for businesses of all sizes.",
    
    "Our flagship product, NOVA, is an AI customer service agent that can handle customer inquiries 24/7 without human intervention.",
    
    "NOVA can be deployed across multiple channels including websites, mobile apps, email, and social media.",
    
    "Nexobotics uses advanced natural language processing to understand and respond to customer queries with human-like understanding.",
    
    "Businesses using Nexobotics solutions report an average of 70% reduction in customer service costs within the first six months.",
    
    "Our AI agents can be customized to match your brand voice and company policies, ensuring consistent customer experiences.",
    
    "Nexobotics offers seamless integration with popular CRM systems including Salesforce, HubSpot, and Zoho.",
    
    "The Nexobotics analytics dashboard provides real-time insights into customer interactions, common issues, and satisfaction metrics.",
    
    "Our AI technology continuously learns from interactions to improve response accuracy and customer satisfaction over time.",
    
    "Nexobotics solutions can handle multiple languages including English, Spanish, French, German, Japanese, and Mandarin.",
    
    "The average response time for queries handled by Nexobotics AI is under 2 seconds, compared to industry average wait times of 11 minutes for human agents.",
    
    "Nexobotics offers both fully automated solutions and hybrid models where AI handles routine queries and escalates complex issues to human agents.",
    
    "Our security protocols ensure all customer data is encrypted and handled in compliance with GDPR, CCPA, and other privacy regulations.",
    
    "Nexobotics was founded in 2020 by a team of AI researchers and customer experience professionals with a mission to transform business-customer relationships.",
    
    "Our subscription plans scale based on query volume, making our solutions accessible to businesses from startups to enterprise corporations."
]

def get_knowledge_data():
    """
    Returns the knowledge base data.
    
    Returns:
        list: A list of strings containing knowledge chunks.
    """
    return KNOWLEDGE_DATA

def add_knowledge_item(item):
    """
    Adds a new item to the knowledge base.
    
    Args:
        item (str): The knowledge item to add.
    
    Returns:
        bool: True if the item was added successfully.
    """
    if item and isinstance(item, str):
        KNOWLEDGE_DATA.append(item)
        return True
    return False

def load_knowledge_from_file(filepath):
    """
    Loads knowledge data from a text file.
    Each line in the file will be treated as a separate knowledge chunk.
    
    Args:
        filepath (str): Path to the text file.
    
    Returns:
        list: The loaded knowledge data.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            new_data = [line.strip() for line in file if line.strip()]
            KNOWLEDGE_DATA.extend(new_data)
        return new_data
    except Exception as e:
        print(f"Error loading knowledge from file: {str(e)}")
        return [] 