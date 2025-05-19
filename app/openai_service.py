import os
import json
from openai import OpenAI
import httpx

# Default system message
DEFAULT_SYSTEM_MESSAGE = """
You are an expert at extracting learning goals from educational documents.
Your task is to identify and extract all learning goals from the provided text.

Each learning goal should be:
1. Clear and concise
2. Action-oriented (use verbs like 'understand', 'explain', 'analyze', etc.)
3. Focused on a single concept or skill

IMPORTANT: Your response MUST be valid JSON with the following structure:
{
  "learning_goals": [
    "First learning goal statement",
    "Second learning goal statement",
    "Third learning goal statement",
    ...
  ]
}

Do not include any explanations or notes outside the JSON structure.
"""

def extract_learning_goals(text, api_key, custom_system_message=None):
    """Extract learning goals from text using OpenAI API"""
    http_client = httpx.Client(
        timeout=60.0,
        follow_redirects=True
    )
    
    client = OpenAI(
        api_key=api_key,
        http_client=http_client,
        base_url="https://api.openai.com/v1"
    )
    
    # Use custom system message if provided, otherwise use default
    system_message = custom_system_message
    
    # Ensure JSON format instructions are always included
    if system_message:
        json_instructions = """
IMPORTANT: Your response MUST be valid JSON with the following structure:
{
  "learning_goals": [
    "First learning goal statement",
    "Second learning goal statement",
    "Third learning goal statement",
    ...
  ]
}

Do not include any explanations or notes outside the JSON structure.
"""
        system_message = f"{system_message}\n\n{json_instructions}"
    else:
        system_message = DEFAULT_SYSTEM_MESSAGE
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Extract learning goals from the following text:\n\n{text[:10000]}"}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        # Extract and parse the response
        result = response.choices[0].message.content
        try:
            # Parse the JSON response
            learning_goals_data = json.loads(result)
            learning_goals = learning_goals_data.get("learning_goals", [])
            if not learning_goals and isinstance(learning_goals_data, list):
                learning_goals = learning_goals_data
        except json.JSONDecodeError:
            # Fallback if the response is not valid JSON
            learning_goals = []
            
        return learning_goals
    
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return [] 