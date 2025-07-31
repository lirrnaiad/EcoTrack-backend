from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import json
import os

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

class AnalyzeRequest(BaseModel):
    company_name: str
    business_type: str

@app.post("/analyze")
async def analyze(data: AnalyzeRequest):
    prompt = f"EcoTrack, ESG breakdown for {data.company_name}, {data.business_type}."

    response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=[
            {
                "role": "system",
                "content": '''
                You are EcoTrack, an AI assistant that provides ESG (Environmental, Social, Governance) analysis of companies.
                
                When a company name is given, your tasks are:
                1. Search for 4–6 recent and relevant public sources (e.g., news, reports, announcements) about the company.
                2. Identify which ESG categories (Environmental, Social, Governance) each source contributes to.
                3. For each relevant category, assign a score from 0 to 100 and briefly explain your reasoning.
                4. Use these individual source contributions to calculate a final ESG score for the company in each category.
                5. Write a short summary of the company’s overall ESG performance.
                            
                Your tone is professional and clear. Your response should be formatted as a JSON object with the following structure:
                
                {
                  "company": "Company Name",
                  "business_type": "If known or inferred, otherwise null",
                  "summary": "Short ESG summary",
                  "esg_scores": {
                    "environmental": {
                      "score": 0–100,
                      "description": "Environmental performance summary"
                    },
                    "social": {
                      "score": 0–100,
                      "description": "Social performance summary"
                    "governance": {
                      "score": 0–100,
                      "description": "Governance performance summary"
                    }
                  },
                  "sources": [
                    {
                      "title": "Source Title",
                      "url": "https://source-link.com",
                      "summary": "Brief article summary",
                      "relevance": ["environmental", "social"],
                      "contributions": {
                        "environmental": {
                          "score": 70,
                          "reasoning": "Highlights sustainability initiatives"
                        },
                        "social": {
                          "score": 60,
                          "reasoning": "Mentions community programs"
                        }
                      }
                    },
                    ...
                  ]
                }
                Only include relevant ESG categories in each source’s contributions. Ensure scores reflect impact and
                reliability of the source.
                '''
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    analysis_text = response.choices[0].message.content
    try:
        analysis = json.loads(analysis_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from AI")

    return JSONResponse(content=analysis)