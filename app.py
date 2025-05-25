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
import requests

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
HF_TOKEN = "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"
NEWSAPI_KEY = "pub_e8e1562f03404268bd119c4044b9ea2d"

client = InferenceClient(token=HF_TOKEN, model="mistralai/Mistral-7B-Instruct-v0.3")
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# ===== EVIDENCE GATHERING ====
def validate_sources(evidence: str, claim: str) -> bool:
    """Ensure sources meet minimum reliability standards"""
    if evidence == "No reliable sources found automatically":
        return False

    trusted_domains = [
        'reuters.com', 'bbc.com', 'apnews.com', 
        'aljazeera.com', 'dawn.com', 'geo.tv',
        'nytimes.com', 'washingtonpost.com'
    ]
    return any(domain in evidence.lower() for domain in trusted_domains)


def fetch_evidence(claim: str) -> str:
    """Retrieve evidence with multiple fallback layers"""
    import requests
    from googlesearch import search
    import wikipedia

    evidence = []

    # Layer 1: NewsData.io API (Primary Source)
    try:
        NEWSAPI_KEY = "pub_e8e1562f03404268bd119c4044b9ea2d"
        response = requests.get(
            "https://newsdata.io/api/1/news",
            params={
                "apikey": NEWSAPI_KEY,
                "q": claim,
                "language": "en",
                "size": 5,
            }
        )
        response.raise_for_status()
        news = response.json()

        evidence.extend([
            f"📰 {article.get('pubDate', 'Unknown')[:10]} | {article.get('source_id', 'Unknown')}: "
            f"{article.get('title', 'No title')}\n{article.get('link', 'No URL')}"
            for article in news.get("results", [])
            if any(d in article.get("link", "").lower() 
                   for d in ['reuters', 'bbc', 'apnews', 'aljazeera', 'cnn'])
        ])
    except Exception as e:
        print(f"NewsData.io Error: {str(e)}")

    # Layer 2: Google Search fallback
    if not evidence:
        try:
            domains = [
                "https://www.reuters.com/search?q=",
                "https://apnews.com/search?q=",
                "https://www.bbc.com/search?q="
            ]
            for domain in domains:
                try:
                    results = search(
                        f"site:{domain.split('//')[-1].split('/')[0]} {claim}",
                        num=3,
                        pause=2.0,
                        stop=3
                    )
                    evidence.extend(results)
                    if evidence:
                        break
                except Exception as search_err:
                    print(f"Search error on {domain}: {search_err}")
        except Exception as e:
            print(f"Direct Search Layer 2 failed: {str(e)}")

    # Layer 3: Wikipedia fallback for historical/political claims
    if not evidence and any(keyword in claim.lower() for keyword in ["elected", "sworn", "appointed"]):
        try:
            wp_summary = wikipedia.summary(claim.split(" in ")[0], sentences=3)
            evidence.append(f"📚 Wikipedia: {wp_summary}")
        except Exception as e:
            print(f"Wikipedia Error: {str(e)}")

    return "\n\n".join(evidence) if evidence else "No reliable sources found automatically"



# ===== MODEL PROCESSING =====
def generate_verdict(claim: str, evidence: str) -> dict:
    """Get model analysis with proper prompt engineering"""
    if evidence == "No reliable sources found":
        # Special handling when no evidence is found
        prompt = f"""<s>[INST] You are a professional fact-checker. Analyze the following claim without external evidence:
Claim: {claim}

Provide structured response:
1. Verdict: [True/False/Misleading/Unverifiable]
2. Confidence: [High/Medium/Low]
3. Summary: [Concise analysis based on your knowledge]
4. Key Evidence: [None]
5. Latest Date: [Unknown] [/INST]"""
    else:
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

    response = client.text_generation(
        prompt=prompt,
        max_new_tokens=250,
        temperature=0.7
    )
    
    # Parse the response
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    result = {
        "verdict": "Unverifiable",
        "confidence": "Low",
        "summary": "Insufficient evidence to verify the claim",
        "key_evidence": "",
        "latest_date": "Unknown"
    }
    
    for line in lines:
        if "Verdict:" in line:
            result["verdict"] = line.split(":")[1].strip()
        elif "Confidence:" in line:
            result["confidence"] = line.split(":")[1].strip()
        elif "Summary:" in line:
            result["summary"] = line.split(":")[1].strip()
        elif "Key Evidence:" in line:
            result["key_evidence"] = line.split(":")[1].strip()
        elif "Latest Date:" in line:
            result["latest_date"] = line.split(":")[1].strip()
    
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
            "sources": evidence,
            "latest_evidence_date": analysis["latest_date"],
            "processing_time": round(time.time() - start_time, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
