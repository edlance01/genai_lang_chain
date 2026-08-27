import os
from dotenv import load_dotenv
from google import genai

# Load variables from the .env file
load_dotenv()

# Step 1: Check if the key is read into Python environment
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print(
        "❌ API Key NOT found in environment! Check your .env file name and location."
    )
else:
    # Obscure key for safe printing
    print(f"✅ Found API Key: {api_key[:6]}...{api_key[-4:]}")

    # Step 2: Validate the key with a quick API call
    try:
        client = (
            genai.Client()
        )  # Automatically picks up GEMINI_API_KEY from environment
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say 'API Key Working!' in 3 words or less.",
        )
        print(f"✅ API Response: {response.text.strip()}")
    except Exception as e:
        print(f"❌ API Call Failed: {e}")
