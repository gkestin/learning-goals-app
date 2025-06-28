import os
import json
from openai import OpenAI
import httpx
from datetime import datetime

# Default system message
DEFAULT_SYSTEM_MESSAGE = """
All learning goals (aka skills) should be in the form:
"<VERB> <OBJECT> <Optional method> <Optional context>"

- Where the verbs are, for example:
Evaluate
Estimate
Interpret
Model (i.e., construct a model)
Classify
Compute
Describe
Visualize
Identify
Determine
Relate (i.e., connect two things)
Locate
Represent
Argue
Justify
Create
Construct
Analyze
Draw
Sketch
Write
Compare
Explain

- Where <Optional method> can be of the form "with ___ formula/technique/method.

- Where  <Optional context> can be of the form "in the context of ___."

NOTE: Restrict learning goals to those that are transferable to other problems or contexts. Omit specifics to the particular problem.

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

DEFAULT_USER_PROMPT = "Extract learning goals from the following text:"

def filter_none_goals(goals):
    """Filter out 'NONE' goals from a list of learning goals"""
    if not goals:
        return []
    return [goal for goal in goals if goal.strip().upper() != "NONE"]

def extract_learning_goals(text, api_key, custom_system_message=None, custom_user_prompt=None, model="gpt-4o", category_title="Default"):
    """Extract learning goals from text using OpenAI API
    
    Args:
        text: Text to extract learning goals from
        api_key: OpenAI API key
        custom_system_message: Custom system prompt (optional)
        custom_user_prompt: Custom user prompt (optional)
        model: OpenAI model to use
        category_title: Title for this category of learning goals
        
    Returns:
        Dict with structured learning goals data for new categorized system
    """
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
    user_prompt = custom_user_prompt or DEFAULT_USER_PROMPT
    
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
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"{user_prompt}\n\n{text[:10000]}"}
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
            
        # Filter out "NONE" goals
        filtered_goals = filter_none_goals(learning_goals)
        
        # Return structured data for new categorized system
        category_data = {
            "goals": filtered_goals,
            "system_prompt": system_message,
            "user_prompt": user_prompt,
            "title": category_title,
            "created_at": datetime.now().isoformat()
        }
        
        return {
            'learning_goals': filtered_goals,  # For backward compatibility
            'category_data': category_data,    # For new categorized system
            'system_message_used': system_message
        }
    
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        category_data = {
            "goals": [],
            "system_prompt": system_message,
            "user_prompt": user_prompt,
            "title": category_title,
            "created_at": datetime.now().isoformat()
        }
        
        return {
            'learning_goals': [],
            'category_data': category_data,
            'system_message_used': system_message
        } 