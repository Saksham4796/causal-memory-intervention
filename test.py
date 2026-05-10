import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_URL") or os.getenv("OPENAI_BASE_URL")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env first.")

client_kwargs = {"api_key": api_key}
if base_url:
    client_kwargs["base_url"] = base_url

client = OpenAI(**client_kwargs)

models = client.models.list()
for m in models.data:
    print(m.id)
