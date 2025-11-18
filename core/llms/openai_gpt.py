import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client from environment variable; fail fast if missing
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not _OPENAI_API_KEY:
    raise RuntimeError(
        "Environment variable OPENAI_API_KEY is not set. Set it to your OpenAI API key."
    )

client = OpenAI(api_key=_OPENAI_API_KEY)
