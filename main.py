import io
import os
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
from fastapi import FastAPI, Header, HTTPException
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
EXPECTED_TOKEN = os.getenv("EXPECTED_TOKEN")

genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

app = FastAPI()

class QARequest(BaseModel):
    documents: str
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

def ask_question(uploaded_uri: str, question: str) -> str:
    response = model.generate_content(
        contents=[{
            "parts": [
                {"file_data": {"file_uri": uploaded_uri, "mime_type": "application/pdf"}},
                {"text": f"Answer in one short sentence: {question}"}
            ]
        }]
    )
    return response.text.strip()

@app.post("/hackrx/run", response_model=QAResponse)
async def run_qa(payload: QARequest, authorization: Optional[str] = Header(None)):
    if authorization != f"Bearer {EXPECTED_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    async with httpx.AsyncClient() as client:
        pdf_response = await client.get(payload.documents)
        if pdf_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download PDF")

        doc_io = io.BytesIO(pdf_response.content)

    uploaded_file = genai.upload_file(doc_io, mime_type="application/pdf")
    uploaded_uri = uploaded_file.uri

    max_workers = min(len(payload.questions), 10)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        answers = list(executor.map(lambda q: ask_question(uploaded_uri, q), payload.questions))

    return {"answers": answers}
