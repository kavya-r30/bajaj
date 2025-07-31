from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Header
from typing import List, Optional
import os
import re
import json
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
EXPECTED_TOKEN = os.getenv("EXPECTED_TOKEN")

model = "mistral-small-latest"
client = Mistral(api_key=MISTRAL_API_KEY)

app = FastAPI()

class QARequest(BaseModel):
    documents: str
    questions: List[str]

def ask_question(uploaded_uri, question):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
Answer the following questions strictly only in single line sentence format less than 25 words. 
Return only a valid JSON array of the answers, in the same order as the questions.
No extra keys, no extra commentary, no extra explanation — just a raw JSON array of strings.

Questions: {question}

Make sure to return valid sentence which is greater than 10 words and less than 25 words and in same order.
"""
                },
                {
                    "type": "document_url",
                    "document_url": uploaded_uri
                }
            ]
        }
    ]

    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )

    match = re.search(r"\[\s*.*?\s*\]", chat_response.choices[0].message.content, re.DOTALL)
    if match:
        cleaned_json_str = match.group(0)
    else:
        raise ValueError("No valid JSON array found in response.")

    return json.loads(cleaned_json_str)

@app.post("/hackrx/run")
async def run_qa(payload: QARequest, authorization: Optional[str] = Header(None)):
    if authorization != f"Bearer {EXPECTED_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    print(f"Doc Link: {payload.documents}")
    print(f"Questions: {payload.questions}")

    answers = ask_question(payload.documents, payload.questions)

    return {"answers": answers}
