"""Quick debug script to test a single Gemini verify call."""
import os, json, asyncio
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

class VerifierResponse(BaseModel):
    ranking: list[int] = Field(description='Segment numbers ranked from most to least relevant.')
    reasoning: str = Field(description='One sentence explanation')

api_key = os.environ['GOOGLE_API_KEY']
with open('dataset/nextgqa/video_upload_cache.json', encoding='utf-8') as f:
    cache = json.load(f)
file_uri = list(cache.values())[0]['file_uri']
print('file_uri:', file_uri)

prompt = (
    'Question: "what did the baby do after throwing the green cup away"\n'
    'Grounding: "The baby throws a green cup away."\n\n'
    'Candidate segments:\n'
    '  Segment 1: 00:02 to 00:27  (2.3s - 27.0s)\n'
    '  Segment 2: 00:00 to 00:27  (0.3s - 27.0s)\n'
    '  Segment 3: 00:02 to 00:14  (2.3s - 14.0s)\n'
    '  Segment 4: 00:02 to 00:14  (2.3s - 14.0s)\n'
    '  Segment 5: 00:18 to 00:27  (18.0s - 27.3s)\n\n'
    'Rank ALL 5 segments from MOST to LEAST relevant.\n'
    'Respond in JSON: {"ranking": [best, ..., worst], "reasoning": "one sentence"}'
)

async def test():
    ac = genai.Client(api_key=api_key).aio

    # Test 1: structured output
    print('\n--- Test 1: structured output ---')
    try:
        resp = await ac.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part(file_data=types.FileData(file_uri=file_uri)),
                types.Part(text=prompt),
            ],
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=VerifierResponse,
            ),
        )
        print('OK:', resp.text[:300])
    except Exception as e:
        print('ERROR:', type(e).__name__, str(e)[:500])

    # Test 2: raw JSON (no schema)
    print('\n--- Test 2: raw JSON ---')
    try:
        resp = await ac.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part(file_data=types.FileData(file_uri=file_uri)),
                types.Part(text=prompt),
            ],
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
            ),
        )
        print('OK:', resp.text[:300])
    except Exception as e:
        print('ERROR:', type(e).__name__, str(e)[:500])

asyncio.run(test())
