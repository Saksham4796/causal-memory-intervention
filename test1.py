import os

from dotenv import load_dotenv
from openai import OpenAI


def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_URL") or os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_AGENT_MODEL", "gpt-4o")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    models = client.models.list()
    print(f"Model list OK. Found {len(models.data)} models.")
    print("First 5 models:", ", ".join(m.id for m in models.data[:5]))

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        temperature=0,
        max_tokens=5,
    )

    print(f"Chat completion OK using model: {model}")
    print("Response:", response.choices[0].message.content)


if __name__ == "__main__":
    main()
