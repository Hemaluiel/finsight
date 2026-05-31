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

Use the following keyword-based rules to assign each transaction to the correct category. Read each transaction description carefully and match it to the FIRST category whose keywords appear in the description (case-insensitive):

1. Groceries         — vegetable, grocery, groceries, market, supermarket, departmental, provisions, sabzi, mandi, bazaar, kirana, food, milk, dairy, fruits, vegetables, meat, fish, poultry
2. Dining & Food     — restaurant, meal, food, snack, dinner, dining, drink, cafe, canteen, lunch, breakfast, bakery, pizza, burger, biryani, sweets, tea stall, dhaba, juice, coffee, bar, pub, alcohol, beverage
3. Transport         — taxi, cab, bus, auto, rickshaw, fuel, petrol, diesel, parking, toll, uber, ola, train, metro, fare, transportation
4. Shopping          — shop, store, mall, boutique, clothing, garments, shoes, fashion, apparel, amazon, flipkart, online order, purchase, ecommerce, retail, electronics, gadget, furniture, home decor, accessories
5. Bills & Utilities — electricity, water, bill, utility, internet, broadband, wifi, gas, lpg, pipeline, sewage, municipal
6. Rent              — rent, house rent, apartment, flat, room rent, monthly rent, accommodation
7. Health & Medical  — hospital, medical, pharmacy, medicine, clinic, doctor, health, diagnostic, lab, nursing, dental, chemist
8. Education         — school, college, university, tuition, fee, course, coaching, library, books, stationery, education, learning, training, workshop, seminar, online class, certification, pencil, notebook, pen, eraser, bag
9. Entertainment     — movie, cinema, game, netflix, spotify, youtube, prime, hotstar, concert, show, event, amusement, fun, recreation, leisure, hobby, music, sports, gym, fitness, dance class, art class, streaming, ticket
10. Subscriptions    — subscription, monthly plan, annual plan, membership, renewal, saas
11. Mobile & Data    — mobile, data, recharge, talk time, sim, prepaid, postpaid, topup, e-load, eight digit numbers starting with 17, 16, 77 (often bill/mobile numbers), provider names like "Airtel", "TashiCell", "BT"
12. Travel           — flight, hotel, resort, travel, trip, tour, booking, airbnb, oyo, holiday, vacation, railway, irctc
13. Personal Care    — salon, hair, spa, cream, shampoo, cosmetic, beauty, grooming, parlour, lotion, deodorant, perfume
14. Contributions & Donations — semso, charity, contribution, donation, rest in peace, rip, fundraiser, relief, temple, church, mosque, offering, zakat
15. Gifts & Celebrations — birthday, gift, celebration, party, marriage, wedding, anniversary, present, function, feast, festival, diwali, christmas, new year, eid, raksha bandhan, valentine's day, mother's day, father's day, greeting card, flowers, cake, gift shop
16. Family & Friends — friend, relative, family, brother, sister, mother, father, parent, cousin, spouse, wife, husband, son, daughter, uncle, aunt, in-law, roommate, partner, buddy, bestie, colleague, neighbor
17. ATM & Cash       — atm, cash withdrawal, cash, teller
18. Insurance        — insurance, lic, policy, premium, cover, insure
19. Investments      — mutual fund, sip, stock, share, investment, fd, fixed deposit, rd, recurring deposit, nps, ppf, gold bond, DK Gold, silver bond, demat, portfolio, capital gain, dividend
20. Loan and Payments     — loan, emi, mortgage, repayment, interest, principal, credit card payment
21. Fund transers       — transfer, fund transfer, sent to, transferred to, received from, payment to,  provides name of a person or entity without any of the above keywords, intrabank transfer, interbank transfer, provider includes bank names like BoB, BNB, BDBL, DK, SBI, HDFC, ICICI, PNB, Axis, Yes Bank, IDFC, Kotak, Union Bank, Canara Bank, etc.
21. Others           — anything that does not match the above categories

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
- Apply keyword rules strictly — do not guess or use generic labels when a keyword matches
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
    # with open("/app/index.html", "r") as f:
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())