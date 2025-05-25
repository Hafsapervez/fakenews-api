import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import requests  # For newsdata.io API
from datetime import datetime, timedelta
import time
from fastapi.middleware.cors import CORSMiddleware

# ===== INITIALIZATION =====
load_dotenv()

app = FastAPI(title="Fact-Checking API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# ===== API KEYS =====
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not NEWSDATA_API_KEY or not HF_TOKEN:
    raise ValueError("Missing API keys in environment variables")

client = InferenceClient(token=HF_TOKEN)

# ===== EVIDENCE GATHERING =====
def fetch_evidence(claim: str) -> str:
    """Fetch evidence from newsdata.io with proper error handling"""
    evidence = []
    
    # 1. newsdata.io API (Primary Source)
    try:
        response = requests.get(
            "https://newsdata.io/api/1/news",
            params={
                "apikey": NEWSDATA_API_KEY,
                "q": f'"{claim}"',
                "language": "en",
                "size": 5
            },
            timeout=10
        )
        
        if response.status_code == 200:
            articles = response.json().get("results", [])
            evidence.extend([
                f"📰 {article.get('pubDate', '')[:10]} | {article.get('source_id', 'Unknown')}: "
                f"{article.get('title', '')}\n{article.get('link', '')}"
                for article in articles
                if any(domain in article.get('link', '').lower()
                      for domain in ['reuters', 'bbc', 'apnews', 'aljazeera'])
            ])
        else:
            print(f"NewsData API Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"NewsData API Exception: {str(e)}")

    # 2. Manual Search Fallback (No Google API to avoid 429 errors)
    if len(evidence) < 3:
        evidence.append("🔍 Manual verification recommended for additional sources")

    return "\n\n".join(evidence) if evidence else "No reliable sources found"

# ===== MODEL PROCESSING =====
def generate_verdict(claim: str, evidence: str) -> dict:
    """Get analysis from Mistral-7B-v0.3"""
    prompt = f"""<s>[INST] You are a fact-checker. Analyze:
Claim: {claim}

Evidence:
{evidence}

Provide response in this format:
1. Verdict: [True/False/Misleading/Unverifiable]
2. Confidence: [High/Medium/Low]
3. Summary: [Concise analysis]
4. Key Evidence: [Most relevant source] [/INST]"""

    try:
        response = client.text_generation(
            prompt=prompt,
            model="mistralai/Mistral-7B-Instruct-v0.3",
            max_new_tokens=200,
            temperature=0.7
        )
        
        # Parse response
        result = {
            "verdict": "Unverifiable",
            "confidence": "Low",
            "summary": "Insufficient evidence",
            "key_evidence": ""
        }
        
        for line in response.split('\n'):
            if "Verdict:" in line:
                result["verdict"] = line.split(":")[1].strip()
            elif "Confidence:" in line:
                result["confidence"] = line.split(":")[1].strip()
            elif "Summary:" in line:
                result["summary"] = line.split(":")[1].strip()
            elif "Key Evidence:" in line:
                result["key_evidence"] = line.split(":")[1].strip()
                
        return result
        
    except Exception as e:
        print(f"Model Error: {str(e)}")
        raise

# ===== API ENDPOINT =====
class ClaimRequest(BaseModel):
    claim: str

class FactCheckResponse(BaseModel):
    verdict: str
    confidence: str
    summary: str
    key_evidence: str
    sources: str
    processing_time: float

@app.post("/factcheck", response_model=FactCheckResponse)
async def factcheck(request: ClaimRequest):
    start_time = time.time()
    
    try:
        evidence = fetch_evidence(request.claim)
        analysis = generate_verdict(request.claim, evidence)
        
        return {
            "verdict": analysis["verdict"],
            "confidence": analysis["confidence"],
            "summary": analysis["summary"],
            "key_evidence": analysis["key_evidence"],
            "sources": evidence,
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
#2nd wokring code
# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from huggingface_hub import InferenceClient
# from newsapi import NewsApiClient
# from googlesearch import search
# import wikipedia
# import time
# from datetime import datetime, timedelta
# from fastapi.middleware.cors import CORSMiddleware

# # ===== INITIALIZATION =====
# load_dotenv()

# app = FastAPI(title="Mistral-7B Fact-Checking API")

# # CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["POST"],
#     allow_headers=["*"],
# )

# ===== SERVICES SETUP =====
# HF_TOKEN = os.getenv("HF_TOKEN")
# GOOGLE_API_KEY ="AIzaSyDVL1OZzaQPsbLKznX5N1TWZo7iX1ewtrY"
# GOOGLE_CSE_ID = "93b82db52b93044b9"
# NEWSAPI_KEY = "17ecec0c43e347f5883259fd4cc06f53"

# if not HF_TOKEN or not NEWSDATA_API_KEY:
#     raise ValueError("Missing required environment variables")

# client = InferenceClient(token=HF_TOKEN, model="mistralai/Mistral-7B-Instruct-v0.3")
# newsapi = NewsApiClient(api_key=NEWSAPI_KEY)  # Removed timeout parameter

# # ===== EVIDENCE GATHERING =====
# def fetch_evidence(claim: str) -> str:
#     """Enhanced evidence gathering with multiple fallbacks"""
#     evidence = []
    
#     # 1. NewsAPI (Primary Source)
#     try:
#         news = newsapi.get_everything(
#             q=f'"{claim}"',
#             language="en",
#             page_size=5,
#             sort_by="publishedAt",
#             from_param=(datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
#         )
#         evidence.extend([
#             f"📰 {article['publishedAt'][:10]} | {article['source']['name']}: "
#             f"{article['title']}\n{article['url']}"
#             for article in news.get('articles', [])
#             if any(d in article['url'].lower() 
#                 for d in ['reuters', 'bbc', 'apnews', 'aljazeera'])
#         ])
#     except Exception as e:
#         print(f"NewsAPI Error: {str(e)}")

#     # 2. Google Search (Fallback)
#     if len(evidence) < 3:  # If less than 3 articles found
#         try:
#             domains = ["reuters.com", "apnews.com", "bbc.com", "aljazeera.com"]
#             query = f'"{claim}" {" OR ".join(f"site:{d}" for d in domains)}'
#             results = list(search(query, num=5, stop=5, pause=2.0))
#             evidence.extend(results[:3])  # Add top 3 results
#         except Exception as e:
#             print(f"Google Search Error: {str(e)}")

#     # 3. Wikipedia (For Historical Facts)
#     if not evidence and any(kw in claim.lower() for kw in ["elected", "sworn", "appointed"]):
#         try:
#             wp_summary = wikipedia.summary(claim.split(" in ")[0], sentences=3)
#             evidence.append(f"📚 Wikipedia: {wp_summary}")
#         except:
#             pass

#     return "\n\n".join(evidence) if evidence else "No reliable sources found"

# # ===== MODEL PROCESSING =====
# def generate_verdict(claim: str, evidence: str) -> dict:
#     """Get model analysis with proper prompt engineering"""
#     prompt = f"""<s>[INST] You are a professional fact-checker. Analyze:
# Claim: {claim}

# Evidence:
# {evidence}

# Provide structured response:
# 1. Verdict: [True/False/Misleading/Unverifiable]
# 2. Confidence: [High/Medium/Low]
# 3. Summary: [Concise analysis]
# 4. Key Evidence: [Most relevant source]
# 5. Latest Date: [YYYY-MM-DD or Unknown] [/INST]"""

#     response = client.text_generation(
#         prompt=prompt,
#         max_new_tokens=250,
#         temperature=0.7
#     )
    
#     # Parse the response
#     lines = [l.strip() for l in response.split('\n') if l.strip()]
#     result = {
#         "verdict": "Unverifiable",
#         "confidence": "Low",
#         "summary": "Insufficient evidence",
#         "key_evidence": "",
#         "latest_date": "Unknown"
#     }
    
#     for line in lines:
#         if "Verdict:" in line:
#             result["verdict"] = line.split(":")[1].strip()
#         elif "Confidence:" in line:
#             result["confidence"] = line.split(":")[1].strip()
#         elif "Summary:" in line:
#             result["summary"] = line.split(":")[1].strip()
#         elif "Key Evidence:" in line:
#             result["key_evidence"] = line.split(":")[1].strip()
#         elif "Latest Date:" in line:
#             result["latest_date"] = line.split(":")[1].strip()
    
#     return result

# # ===== API ENDPOINTS =====
# class ClaimRequest(BaseModel):
#     claim: str

# class FactCheckResponse(BaseModel):
#     verdict: str
#     confidence: str
#     summary: str
#     key_evidence: str
#     sources: str
#     latest_evidence_date: str
#     processing_time: float

# @app.post("/factcheck", response_model=FactCheckResponse)
# async def factcheck(request: ClaimRequest):
#     start_time = time.time()
    
#     try:
#         # Step 1: Gather Evidence
#         evidence = fetch_evidence(request.claim)
        
#         # Step 2: Generate Verdict
#         analysis = generate_verdict(request.claim, evidence)
        
#         return {
#             "verdict": analysis["verdict"],
#             "confidence": analysis["confidence"],
#             "summary": analysis["summary"],
#             "key_evidence": analysis["key_evidence"],
#             "sources": evidence,
#             "latest_evidence_date": analysis["latest_date"],
#             "processing_time": round(time.time() - start_time, 2)
#         }
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Fact-checking failed: {str(e)}"
#         )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

#3rd working code
# ===== INSTALLATIONS =====
# !pip install fastapi uvicorn transformers accelerate bitsandbytes pyngrok google-api-python-client wikipedia python-dotenv newsapi-python

# ===== IMPORTS =====
# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from huggingface_hub import login
# import torch
# import uvicorn
# from pyngrok import ngrok
# import nest_asyncio
# from fastapi.middleware.cors import CORSMiddleware
# from googlesearch import search
# import wikipedia
# from newsapi import NewsApiClient
# import time
# from datetime import datetime, timedelta

# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
# MAX_TOKENS = 150  # Reduced for Colab stability

# # ===== INITIALIZATION =====
# load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
# NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# # Validate they exist
# if not HF_TOKEN:
#     raise ValueError("HF_TOKEN environment variable not set!")
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY environment variable not set!")
# if not NEWSAPI_KEY:
#     raise ValueError("NEWSAPI_KEY environment variable not set!")
# if not GOOGLE_CSE_ID:
#     raise ValueError("GOOGLE_CSE_ID environment variable not set!")

# login(token=HF_TOKEN)
# app = FastAPI(title="Mistral-7B Fact-Checking API")
# # CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["POST"],
#     allow_headers=["*"],
# )
# # ===== MODEL LOADING =====
# print("⚙️ Loading model...")
# try:
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         device_map="auto",
#         torch_dtype=torch.float16,
#         load_in_4bit=True,
#         token=HF_TOKEN
#     )
#     print("✅ Model loaded successfully!")
# except Exception as e:
#     print(f"❌ Model loading failed: {str(e)}")
#     raise

# # ===== RAG FUNCTIONS =====
# def validate_sources(evidence: str, claim: str) -> bool:
#     """Ensure sources meet minimum reliability standards"""
#     if evidence == "No reliable sources found automatically":
#         return False
        
#     # Check if sources contain at least one trusted domain
#     trusted_domains = [
#         'reuters.com', 'bbc.com', 'apnews.com', 
#         'aljazeera.com', 'dawn.com', 'geo.tv',
#         'nytimes.com', 'washingtonpost.com'
#     ]
#     return any(domain in evidence.lower() for domain in trusted_domains)

# def fetch_evidence(claim: str) -> str:
#     """Retrieve evidence with multiple fallback layers"""
#     evidence = []
    
#     # Layer 1: NewsAPI with extended timeout
#     try:
#         newsapi = NewsApiClient(api_key=NEWSAPI_KEY, timeout=10)  # Increased timeout
#         news = newsapi.get_everything(
#             q=claim,
#             sources="bbc-news,reuters,al-jazeera-english,associated-press",
#             language="en",
#             page_size=5,
#             sort_by="publishedAt",
#             from_param=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
#         )
#         evidence.extend([
#             f"📰 {article['publishedAt'][:10]} | {article['source']['name']}: "
#             f"{article['title']}\n{article['url']}"
#             for article in news.get('articles', [])
#         ])
#     except Exception as e:
#         print(f"NewsAPI Layer 1 failed: {str(e)}")

#     # Layer 2: Direct domain scraping if NewsAPI fails
#     if not evidence:
#         try:
#             domains = [
#                 "https://www.reuters.com/search?q=",
#                 "https://apnews.com/search?q=",
#                 "https://www.bbc.com/search?q="
#             ]
#             for domain in domains:
#                 try:
#                     results = search(
#                         f"site:{domain.split('//')[-1].split('/')[0]} {claim}",
#                         num=3,
#                         pause=2.0,
#                         stop=3
#                     )
#                     evidence.extend(results)
#                     if evidence: break
#                 except:
#                     continue
#         except Exception as e:
#             print(f"Direct Search Layer 2 failed: {str(e)}")

#     # Layer 3: Wikipedia for historical facts
#     if not evidence and any(keyword in claim.lower() for keyword in ["elected", "sworn", "appointed"]):
#         try:
#             wp_summary = wikipedia.summary(claim.split(" in ")[0], sentences=3)
#             evidence.append(f"📚 Wikipedia: {wp_summary}")
#         except:
#             pass

#     return "\n\n".join(evidence) if evidence else "No reliable sources found automatically"

# def parse_model_response(response: str) -> dict:
#     """Parse and validate the model's structured response"""
#     response = response.replace("Verdict: ", "").replace("Summary: ", "")
    
#     result = {
#         "verdict": "Unverifiable",
#         "confidence": "Low",
#         "summary": "Insufficient evidence to verify this claim",
#         "key_evidence": "",
#         "latest_evidence_date": "Unknown"
#     }
    
#     try:
#         lines = [line.strip() for line in response.split("\n") if line.strip()]
#         for line in lines:
#             if line.lower().startswith("1.") or "verdict" in line.lower():
#                 result["verdict"] = line.split(":")[-1].strip()
#             elif line.lower().startswith("2.") or "confidence" in line.lower():
#                 result["confidence"] = line.split(":")[-1].strip()
#             elif line.lower().startswith("3.") or "summary" in line.lower():
#                 result["summary"] = line.split(":")[-1].strip()
#             elif line.lower().startswith("4.") or "evidence" in line.lower():
#                 result["key_evidence"] = line.split(":")[-1].strip()
#             elif line.lower().startswith("5.") or "date" in line.lower():
#                 result["latest_evidence_date"] = line.split(":")[-1].strip()
#     except Exception as e:
#         print(f"Parsing error: {str(e)}")
    
#     return result

# # ===== API ENDPOINTS =====
# class ClaimRequest(BaseModel):
#     claim: str

# class FactCheckResponse(BaseModel):
#     verdict: str
#     confidence: str
#     summary: str
#     key_evidence: str
#     sources: str
#     latest_evidence_date: str
#     processing_time: float

# @app.post("/factcheck", response_model=FactCheckResponse)
# async def factcheck(request: ClaimRequest):
#     start_time = time.time()
#     evidence = fetch_evidence(request.claim)
    
#     # Critical check before processing
#     if not validate_sources(evidence, request.claim):
#         return {
#             "verdict": "Unverifiable",
#             "confidence": "Low",
#             "summary": "No reliable sources could be found to verify this claim",
#             "key_evidence": "",
#             "sources": evidence,
#             "latest_evidence_date": "Unknown",
#             "processing_time": round(time.time() - start_time, 2)
#         }
    
#     try:
#         # Step 2: Generate fact-check
#         prompt = f"""<s>[INST] You are a fact-checking assistant. Analyze this claim:
# Claim: {request.claim}

# Available Evidence:
# {evidence}

# Provide your analysis in this EXACT format:
# 1. Verdict (True/False/Misleading/Unverifiable)
# 2. Confidence (High/Medium/Low)
# 3. Summary (Concise factual summary)
# 4. Key Evidence (Most relevant source)
# 5. Date of Latest Evidence (YYYY-MM-DD or 'Unknown')

# Important guidelines:
# - Prefer recent evidence (last 3 months)
# - If no evidence found, verdict should be 'Unverifiable'
# - For country-specific claims, prioritize local sources [/INST]"""
        
#         inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=MAX_TOKENS,
#             temperature=0.7,
#             do_sample=True
#         )
        
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         response = response.split("[/INST]")[-1].strip()
#         parsed_response = parse_model_response(response)

#         return {
#             "verdict": parsed_response["verdict"],
#             "confidence": parsed_response["confidence"],
#             "summary": parsed_response["summary"],
#             "key_evidence": parsed_response["key_evidence"],
#             "sources": evidence,
#             "latest_evidence_date": parsed_response["latest_evidence_date"],
#             "processing_time": round(time.time() - start_time, 2)
#         }
#     except torch.cuda.OutOfMemoryError:
#         raise HTTPException(
#             status_code=500,
#             detail="GPU memory exceeded - try reducing MAX_TOKENS"
#         )
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Processing error: {str(e)}"
#         )

# # ===== SERVER LAUNCH =====
# if __name__ == "__main__":
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
