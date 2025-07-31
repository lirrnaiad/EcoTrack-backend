from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


class AnalyzeRequest(BaseModel):
    company_name: str
    business_type: str

@app.get("/")
async def root():
    return {"message": "EcoTrack ESG Analyzer API"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze(data: AnalyzeRequest):
    try:
        prompt = f"EcoTrack, ESG breakdown for {data.company_name}, {data.business_type}."

        response = client.chat.completions.create(
            model='gpt-4o-mini',  # Fixed model name
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
                    5. Write a short summary of the company's overall ESG performance.

                    Your tone is professional and clear. Your response should be formatted as a JSON object with the following structure:

                    {
                      "company": "Company Name",
                      "business_type": "If known or inferred, otherwise null",
                      "summary": "Short ESG summary",
                      "esg_scores": {
                        "environmental": {
                          "score": 0,
                          "description": "Environmental performance summary"
                        },
                        "social": {
                          "score": 0,
                          "description": "Social performance summary"
                        },
                        "governance": {
                          "score": 0,
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
                        }
                      ]
                    }
                    Only include relevant ESG categories in each source's contributions. Ensure scores reflect impact and
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

        # Clean the response if it contains markdown code blocks
        if analysis_text.startswith('```json'):
            analysis_text = analysis_text.strip('```json').strip('```').strip()

        analysis = json.loads(analysis_text)

        return JSONResponse(content=analysis)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON response from AI: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.exception_handler(405)
async def method_not_allowed_handler(request, exc):
    return JSONResponse(
        status_code=405,
        content={"detail": f"Method {request.method} not allowed for {request.url.path}"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))