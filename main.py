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
        prompt = f"""The company is: {data.company_name}
                    Business type: {data.business_type or 'Unknown'}

                    Your tasks:
                    1. Identify 1-5 recent, real, relevant sources about this company (news, reports, announcements).
                       - Each must have: title, URL, publisher, publish_date, and a short summary.
                       - Skip duplicates or irrelevant results.
                    2. For each source, decide which ESG categories (Environmental, Social, Governance) it informs.
                    3. For each relevant category, assign a score from 0–100 and provide one-sentence reasoning tied directly to the text.
                    4. Aggregate the source-level contributions into final ESG scores (E, S, G), balancing credibility and recency.
                    5. Write a short summary of the company's overall ESG performance.

                    Output JSON structure:
                    {{
                      "company": "{data.company_name}",
                      "business_type": "{data.business_type or 'Unknown'}",
                      "summary": "Short ESG summary",
                      "esg_scores": {{
                        "environmental": {{
                          "score": 0,
                          "description": "Environmental performance summary"
                        }},
                        "social": {{
                          "score": 0,
                          "description": "Social performance summary"
                        }},
                        "governance": {{
                          "score": 0,
                          "description": "Governance performance summary"
                        }}
                      }},
                      "sources": [
                        {{
                          "title": "Source Title",
                          "url": "https://source-link.com",
                          "publisher": "Publisher Name",
                          "published_date": "YYYY-MM-DD",
                          "summary": "Brief article summary",
                          "relevance": ["environmental", "social"],
                          "contributions": {{
                            "environmental": {{
                              "score": 70,
                              "reasoning": "Highlights new waste reduction targets"
                            }},
                            "social": {{
                              "score": 55,
                              "reasoning": "Mentions employee welfare programs"
                            }}
                          }}
                        }}
                      ]
                    }}

                    Constraints:
                    - If no valid sources are found for a pillar, leave that pillar's score 0 and description "Insufficient evidence."
                    - Do not include categories not mentioned in the source.
                    - Keep total sources between 4–6.
                    - All reasoning must be grounded in actual text from the source."""

        response = client.chat.completions.create(
            model='gpt-5-mini-2025-08-07',
            messages=[
                {
                    "role": "system",
                    "content": """You are EcoTrack, an AI assistant built for BPI to evaluate SMEs on ESG (Environmental, Social, and Governance).

                    You must only use real, verifiable sources. Never invent or fabricate URLs, titles, or publishers.
                    If no source is found, clearly say "insufficient evidence."

                    Rules:
                    - Cite only from valid URLs that can be retrieved. Do not generate imaginary links.
                    - If a source is not credible (e.g., anonymous blog, broken link), omit it.
                    - Abstain when evidence is insufficient; do not guess.
                    - Each ESG score must be traceable to specific sources.
                    - Prefer recent sources (within last 36 months) and credible domains (news, gov, NGO, reputable business media).
                    - Keep tone professional, concise, and neutral.
                    - Output strictly in JSON (no extra commentary)."""
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