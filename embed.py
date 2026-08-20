from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"


def create_embedding(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding