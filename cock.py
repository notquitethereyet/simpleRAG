import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_search_query(user_input):
    """
    Use Groq to infer a concise search query.
    :param user_input: String input from the user.
    :return: String search query.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Add explicit instructions for a concise and focused query
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Use this data and answer the query consiely: {user_input}"
            }
        ],
        model="llama-3.3-70b-versatile"
    )

    # Extract and return the refined query
    return response.choices[0].message.content.strip()
