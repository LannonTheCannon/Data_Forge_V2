import OpenAI
import os


client = OpenAI(api_token=os.environ["OPENAI_API_KEY"])
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "give me a analytical break down of this chart"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "http://localhost:8501/media/7df64cfb7976d04460fb43aa5bbf88c79cf6fe0c09ed973cc8fdaddc.png",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0])