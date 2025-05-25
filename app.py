import os 
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from newsapi import NewsApiClient
from googlesearch import search
import wikipedia
import time
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware

# ===== INITIALIZATION =====
load_dotenv()

app = FastAPI(title="Mistral-7B Fact-Checking API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# ===== SERVICES SETUP =====
HF_TOKEN = os.getenv("HF_TOKEN")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")  # Correctly loaded from environment

if not HF_TOKEN or not NEWSAPI_KEY:
    raise ValueError("Missing required environment variables: HF_TOKEN or NEWSAPI_KEY")

client = InferenceClient(token=HF_TOKEN, model="mistralai/Mistral-7B-Instruct-v0.3")
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# ===== EVIDENCE GATHERING =====
def fetch_evidence(claim: str) -> str:
    """Enhanced evidence gathering with multiple fallbacks"""
    evidence = []

    # 1. NewsAPI (Primary Source)
    try:
        news = newsapi.get_everything(
            q=f'"{claim}"',
            language="en",
            page_size=5,
            sort_by="publishedAt",
            from_param=(datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        )
        evidence.extend([
            f"📰 {article['publishedAt'][:10]} | {article['source']['name']}: "
            f"{article['title']}\n{article['url']}"
            for article in news.get('articles', [])
            if any(d in article['url'].lower() 
                   for d in ['reuters', 'bbc', 'apnews', 'aljazeera'])
        ])
    except Exception as e:
        print(f"NewsAPI Error: {str(e)}")

    # 2. Google Search (Fallback)
    if len(evidence) < 3:
        try:
            domains = ["reuters.com", "apnews.com", "bbc.com", "aljazeera.com", "cnn.com"]
            query = f'"{claim}" {" OR ".join(f"site:{d}" for d in domains)}'
            results = list(search(query, num=5, stop=5, pause=2.0))
            evidence.extend(results[:3])
        except Exception as e:
            print(f"Google Search Error: {str(e)}")

    # 3. Wikipedia (For Historical/Political Claims)
    if not evidence and any(kw in claim.lower() for kw in ["elected", "sworn", "appointed"]):
        try:
            wp_summary = wikipedia.summary(claim.split(" in ")[0], sentences=3)
            evidence.append(f"📚 Wikipedia: {wp_summary}")
        except:
            pass

    return "\n\n".join(evidence) if evidence else ""

# ===== MODEL PROCESSING =====
def generate_verdict(claim: str, evidence: str) -> dict:
    """Return model analysis with or without RAG"""
    if evidence:
        # Use RAG
        prompt = f"""<s>[INST] You are a professional fact-checker. Analyze:
Claim: {claim}

Evidence:
{evidence}

Provide structured response:
1. Verdict: [True/False/Misleading/Unverifiable]
2. Confidence: [High/Medium/Low]
3. Summary: [Concise analysis]
4. Key Evidence: [Most relevant source]
5. Latest Date: [YYYY-MM-DD or Unknown] [/INST]"""
    else:
        # Use only LLM generation with no evidence (when nothing verifiable is found)
        prompt = f"""<s>[INST] You are a professional fact-checker. A user submitted the following claim:
Claim: {claim}

There are no reliable or verifiable sources found in news or search engines.

Please analyze the statement based on your general world knowledge and common-sense reasoning.

Provide structured response:
1. Verdict: [True/False/Misleading/Unverifiable]
2. Confidence: [High/Medium/Low]
3. Summary: [Concise explanation of why]
4. Key Evidence: [LLM-based reasoning]
5. Latest Date: [Unknown or N/A] [/INST]"""

    response = client.text_generation(
        prompt=prompt,
        max_new_tokens=250,
        temperature=0.7
    )

    # Parse the response
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    result = {
        "verdict": "False",
        "confidence": "Low",
        "summary": "Insufficient evidence",
        "key_evidence": "",
        "latest_date": "Unknown"
    }

    for line in lines:
        if "Verdict:" in line:
            result["verdict"] = line.split(":", 1)[1].strip()
        elif "Confidence:" in line:
            result["confidence"] = line.split(":", 1)[1].strip()
        elif "Summary:" in line:
            result["summary"] = line.split(":", 1)[1].strip()
        elif "Key Evidence:" in line:
            result["key_evidence"] = line.split(":", 1)[1].strip()
        elif "Latest Date:" in line:
            result["latest_date"] = line.split(":", 1)[1].strip()

    return result

# ===== API ENDPOINTS =====
class ClaimRequest(BaseModel):
    claim: str

class FactCheckResponse(BaseModel):
    verdict: str
    confidence: str
    summary: str
    key_evidence: str
    sources: str
    latest_evidence_date: str
    processing_time: float

@app.post("/factcheck", response_model=FactCheckResponse)
async def factcheck(request: ClaimRequest):
    start_time = time.time()

    try:
        # Step 1: Gather Evidence
        evidence = fetch_evidence(request.claim)

        # Step 2: Generate Verdict
        analysis = generate_verdict(request.claim, evidence)

        return {
            "verdict": analysis["verdict"],
            "confidence": analysis["confidence"],
            "summary": analysis["summary"],
            "key_evidence": analysis["key_evidence"],
            "sources": evidence or "No reliable sources found.",
            "latest_evidence_date": analysis["latest_date"],
            "processing_time": round(time.time() - start_time, 2)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fact-checking failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
