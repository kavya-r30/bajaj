import os
import re
import json
import requests
from typing import List
from agno.tools import tool
from agno.agent import Agent
from bs4 import BeautifulSoup
from mistralai import Mistral
from pydantic import BaseModel
from dotenv import load_dotenv
from agno.models.mistral import MistralChat
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Header, HTTPException, status, FastAPI
import uvicorn

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("MISTRAL_API_KEY")
VALID_AUTH_KEY = os.getenv("EXPECTED_TOKEN")

@tool
def CityPuzzle(doc_url : str) -> List[str]:
    """
    Solves a multi-step PDF puzzle to determine a secret flight number.

    This function extracts structured data from a given PDF document,
    uses it to look up a favorite city, maps it to a landmark, then maps
    the landmark to a flight number URL, and finally retrieves the flight number.

    Args:
        doc_url (str): The URL of the PDF document containing the puzzle data.

    Returns:
        List[str]: 
            - On success: A list containing the secret flight number as a single string.
            - On failure: A list containing an error message string.
    """
    def func():
        model = "mistral-small-latest"
        client = Mistral(api_key=api_key)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                        Given the context from the provided document, generate a single, valid JSON object. This object must not be enclosed in markdown backticks. The root of this JSON object must contain exactly three keys: `favorite_city_link`, `landmark_to_flight_link`, and `city_to_landmark_mapping`.
                        - The value for the `favorite_city_link` key must be the string URL provided in the document for retrieving the favorite city.
                        - The value for the `landmark_to_flight_link` key must be a JSON object that maps specific landmarks to their corresponding flight number URLs as detailed in the document. This object should include a key named `other_landmarks` for the generic flight number URL.
                        - The value for the `city_to_landmark_mapping` key must be a JSON object. The keys of this object should be the 'Current Location' cities from all the tables in the document, and their corresponding values should be the 'Landmark' found in that city. If a city appears more than once with different landmarks, the value for that city should be an array of all its associated landmarks.
                        """
                    },
                    {
                        "type": "document_url",
                        "document_url": doc_url
                    }
                ]
            }
        ]
        chat_response = client.chat.complete(model=model, messages=messages)
        content = chat_response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in the API response.")
        json_string = json_match.group(0)
        return json.loads(json_string)

    def solve_ctf():
        mappings = func()
        if not mappings:
            return ["Error: Could not retrieve data mappings."]
        try:
            city_url = mappings.get("favorite_city_link")
            response = requests.get(city_url, timeout=10)
            response.raise_for_status()
            favorite_city = response.json().get("data", {}).get("city")
            if not favorite_city:
                return ["Error: Could not find 'city' in the API response."]
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            return [f"Error fetching favorite city: {e}"]

        landmark = mappings.get("city_to_landmark_mapping", {}).get(favorite_city)
        if not landmark:
            return [f"Error: City '{favorite_city}' not found in map."]
        if isinstance(landmark, list):
            landmark = landmark[0]

        landmark_map = mappings.get("landmark_to_flight_link", {})
        final_url = landmark_map.get(landmark, landmark_map.get("other_landmarks"))
        if not final_url:
            return ["Error: Could not determine final flight URL."]

        try:
            final_response = requests.get(final_url, timeout=10)
            final_response.raise_for_status()
            flight_number = final_response.json().get("data", {}).get("flightNumber")
            if flight_number:
                print(f"Flight number: {flight_number}")
                return [flight_number]
            else:
                return ["Error: 'flightNumber' key not found in final response."]
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            return [f"Error fetching final flight number: {e}"]

    return solve_ctf()

@tool
def GetToken(url: str) -> List[str]:
    """
    Retrieves a secret token from a webpage.

    This function fetches the HTML content from the given URL, searches for a 
    <div> element with `id="token"`, and extracts its text.

    Args:
        url (str): The URL of the webpage containing the token.

    Returns:
        List[str]:
            - On success: A list containing the extracted token as a single string.
            - On failure: A list containing an error message string.
    """

    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        token_div = soup.find('div', id='token')
        if token_div:
            token_value = token_div.get_text(strip=True)
            print(f"Extracted token: {token_value}")
            return [token_value]
        else:
            return ["Token div not found in the HTML"]
    except requests.RequestException as e:
        return [f"Error fetching the URL: {e}"]
    except Exception as e:
        return [f"Error parsing HTML: {e}"]

@tool
def MRag(questions: List[str], document_url : str) -> List[str]:
    """
    Answers questions based on the context of a bilingual (Malayalam and English) PDF news article.

    Uses the provided PDF to answer each question in a single sentence of 
    10-25 words in the same language as the PDF context (not the question).
    Stick to the pdf context and Return the response from the given context only.

    Args:
        questions (List[str]): A list of questions to be answered.
        document_url (str): The URL of the PDF document to use for context.

    Returns:
        List[str]: 
            A list of answers in the same order as the questions.
            Each answer is a single sentence string.
            If no context is found, the string "no context given" is used.
    """

    model = "mistral-small-latest"
    client = Mistral(api_key=api_key)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
                    Answer the following questions strictly only in single line sentence format less than 25 words in the language given of the document context not question.
                    Return only a valid JSON array of the answers, in the same order as the questions.
                    Please stick to the pdf context and return the response from the given context only.
                    No extra keys, no extra commentary, no extra explanation, just a raw JSON array of strings.

                    Questions: {questions}

                    Make sure to return valid sentence which is greater than 10 words and less than 25 words and in same order
                    """
                },
                {
                    "type": "document_url",
                    "document_url": document_url
                }
            ]
        }
    ]

    chat_response = client.chat.complete(model=model, messages=messages)
    content = chat_response.choices[0].message.content
    match = re.search(r'\[.*\]', content, re.DOTALL)
    if match:
        cleaned_json_str = match.group(0)
        try:
            print("Answer: ", cleaned_json_str)
            return json.loads(cleaned_json_str)
        except json.JSONDecodeError:
            return ["Error: Failed to parse the returned JSON array."]
    return ["Error: Could not find a JSON array in the response."]

class ResponseModel(BaseModel):
    answers: List[str]

hackrx_agent = Agent(
    model=MistralChat(id="mistral-small-latest", api_key=api_key),
    name="HackRx Challenge Solver",
    description="A specialized agent designed to solve the multi-stage HackRx challenge. It can analyze different inputs (PDFs, web URLs) and questions to perform tasks like solving complex puzzles, extracting specific data from a webpage, or answering questions about a document.",
    instructions="""
    **Your Goal:** Solve one of three possible challenges based on the user's input.

    **Step 1: Analyze the Input**
    Carefully examine the user's prompt, which will contain a URL and a list of questions. The combination of these two elements is the key to your decision.

    **Step 2: Apply Routing Logic**
    You MUST choose only ONE tool per request. Use the following logic to decide:
    - **Use `CityPuzzle` IF:** The URL contains `FinalRound4SubmissionPDF.pdf` AND the question is about finding a "flight number". This tool takes no arguments.
    - **Use `GetToken` IF:** The URL contains `get-secret-token` AND the question is about getting a "secret token". You MUST pass the URL from the prompt to this tool.
    - **Use `MRag` IF:** The URL contains `pdf` AND the prompt contains a list of multiple questions (especially if some are in a foreign language like Malayalam or English). You MUST pass the list of questions from the prompt to this tool.

    **Step 3: Format the Final Output**
    Execute the chosen tool and return its output. Your final response MUST conform to the provided `response_model`, which expects a JSON object with a single key "answers" that contains a list of strings. All tools are designed to return a `List[str]`, so their output can be directly used.
    """,
    tools=[CityPuzzle, MRag, GetToken],
    response_model=ResponseModel,
    success_criteria="The agent has succeeded if it correctly identifies the challenge type, executes the appropriate tool with the correct arguments, and returns the result (flight number, token, or a list of answers) formatted correctly within the `response_model`."
)

class ChallengeRequest(BaseModel):
    documents: str
    questions: List[str]

@app.get('/health')
async def health_check():
    return {"status": "healthy", "message": "API is running"}

@app.post("/hackrx/run", response_model=ResponseModel)
async def solve_challenge_endpoint(request: ChallengeRequest, authorization: str = Header(None)):
    """
    Receives a challenge URL and a list of questions.
    Requires a Bearer token in the Authorization header.
    Routes the request to an AI agent that selects the appropriate tool to find the solution.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing or improperly formatted."
        )

    token = authorization.split("Bearer ")[1].strip()
    if token != VALID_AUTH_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Bearer token."
        )
    
    print(f"URL: {request.documents}")
    print(f"Questions: {request.questions}")

    agent_response = hackrx_agent.run(f"The URL is : {request.documents} and the Question(s) are : {request.questions}")

    return agent_response.content

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


