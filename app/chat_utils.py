# app/chat_utils.py

import os
from google import genai

# ---------------------------
# Setup Gemini Client
# ---------------------------
def get_genai_client():
    """
    Returns a configured Gemini client
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key)

# ---------------------------
# Generate response
# ---------------------------
def generate_response(system_prompt: str, user_prompt: str = None) -> str:
    """
    Generate a response from Google Gemini using the new SDK.
    """
    client = get_genai_client()

    # Combine prompts
    prompt = system_prompt
    if user_prompt:
        prompt += f"\nUser: {user_prompt}"

    # Call Gemini model
    response = client.models.generate_content(
        model="gemini-2.5-flash",  # or "gemini-2.5-mini" for faster responses
        contents=prompt,
    )

    # The generated text is in response.text
    return response.text
