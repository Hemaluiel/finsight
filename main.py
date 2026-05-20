"""
Bank Statement Analyzer - FastAPI Backend (Groq)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from groq import Groq
import pdfplumber
import json
import os
import io
from datetime import datetime
from contextlib import asynccontextmanager

#  Config 
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-key-here")
client = Groq(api_key=GROQ_API_KEY)

# App 
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="Bank Statement Analyzer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Analyze endpoint
@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 20MB)")

    # Extract text 
    try:
        text = ""
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as e:
        raise HTTPException(500, f"PDF parsing failed: {str(e)}")

    if len(text.strip()) < 50:
        raise HTTPException(400, "Could not extract text. Please use a text-based PDF (not a scanned image).")

    # Groq AI analysis 
    prompt = f"""You are an expert bank statement analyzer. Analyze the following bank statement and identify ALL debit/expense transactions.

Categorize them into groups such as: Food & Dining, Groceries, Transport, Utilities, Shopping, Entertainment, Healthcare, Education, Rent/Housing, Insurance, Subscriptions, ATM Withdrawals, Transfers, Fuel, Travel, Other.

Return ONLY valid JSON, no markdown, no extra text:
{{
  "categories": [
    {{ "name": "Category Name", "amount": 1234.56, "count": 5, "percentage": 23.4 }},
    ...
  ],
  "top5": [
    {{ "description": "Merchant/description", "amount": 123.45, "category": "Category", "date": "DD/MM or as shown" }},
    ...
  ],
  "total_debits": 9999.99,
  "total_credits": 8888.88,
  "currency": "symbol or code e.g. $ or INR or Nu.",
  "period": "Month Year or date range if detectable",
  "insights": "2-3 sentence summary of spending patterns and notable observations"
}}

Rules:
- Sort categories by amount descending
- Top 5 = 5 largest individual debit transactions
- Only include categories with amount > 0
- Percentages must add up to 100

Bank statement text (first 14000 chars):
{text[:14000]}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(500, "AI returned invalid JSON. Please try again.")
    except Exception as e:
        raise HTTPException(500, f"AI analysis failed: {str(e)}")

    result["filename"] = file.filename
    result["analyzed_at"] = datetime.utcnow().isoformat()

    return result

# Health 
@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# Serve frontend 
@app.get("/")
async def serve_index():
    with open("/app/index.html", "r") as f:
        return HTMLResponse(content=f.read())