from dotenv import load_dotenv
from openai import OpenAI

# Load our secret API key
load_dotenv()

# Create the OpenAI client
client = OpenAI()

# Send a question to the LLM
response = client.responses.create(
    model="gpt-5.5",
    input="Explain RAG in three simple sentences."
)

# Display the AI response
print(response.output_text)
