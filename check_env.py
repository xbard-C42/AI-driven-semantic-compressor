# check_env.py
from dotenv import load_dotenv
import os

# Load and override from .env in the repo root
load_dotenv(dotenv_path=".env", override=True)

print("OPENAI_API_KEY:    ", os.getenv("OPENAI_API_KEY"))
print("ANTHROPIC_API_KEY: ", os.getenv("ANTHROPIC_API_KEY"))
print("GOOGLE_API_KEY:    ", os.getenv("GOOGLE_API_KEY"))
