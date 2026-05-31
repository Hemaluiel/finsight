"""
Bank Statement Analyzer — Netlify Serverless Function
Replaces the FastAPI /api/analyze endpoint.
"""

import base64
import io
import json
import os
import re
from datetime import datetime

import pdfplumber
from groq import Groq

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(status: int, message: str) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        "body": json.dumps({"detail": message}),
    }


def _parse_multipart(body_bytes: bytes, content_type: str):
    """
    Minimal multipart/form-data parser.
    Returns (filename, file_bytes) for the first file part, or (None, None).
    """
    # Extract boundary from Content-Type header
    boundary_match = re.search(r'boundary=([^\s;]+)', content_type)
    if not boundary_match:
        return None, None

    boundary = boundary_match.group(1).strip('"').encode()
    parts = body_bytes.split(b'--' + boundary)

    for part in parts:
        if b'filename=' not in part:
            continue

        # Split headers from body on the first blank line (CRLF CRLF)
        separator = part.find(b'\r\n\r\n')
        if separator == -1:
            continue

        raw_headers = part[:separator]
        file_bytes = part[separator + 4:]

        # Strip the trailing CRLF added by multipart framing
        if file_bytes.endswith(b'\r\n'):
            file_bytes = file_bytes[:-2]

        # Pull filename out of Content-Disposition header
        fname_match = re.search(rb'filename="?([^"\r\n;]+)"?', raw_headers)
        filename = fname_match.group(1).decode(errors="replace") if fname_match else "upload.pdf"

        return filename, file_bytes

    return None, None


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

def handler(event, context):
    # --- CORS pre-flight ---
    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
            "body": "",
        }

    if event.get("httpMethod") != "POST":
        return _error(405, "Method Not Allowed")

    # --- Decode body ---
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body_bytes = base64.b64decode(body)
    else:
        body_bytes = body.encode("latin-1")   # latin-1 preserves raw bytes

    content_type = (event.get("headers") or {}).get("content-type", "")

    # --- Parse uploaded file ---
    filename, pdf_bytes = _parse_multipart(body_bytes, content_type)

    if not pdf_bytes:
        return _error(400, "No file found in request. Send a multipart/form-data POST with field name 'file'.")

    if not filename.lower().endswith(".pdf"):
        return _error(400, "Only PDF files are accepted.")

    if len(pdf_bytes) > 20 * 1024 * 1024:
        return _error(400, "File too large (max 20 MB).")

    # --- Extract text from PDF ---
    try:
        text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as exc:
        return _error(500, f"PDF parsing failed: {exc}")

    if len(text.strip()) < 50:
        return _error(
            400,
            "Could not extract text. Please use a text-based PDF (not a scanned image).",
        )

    # --- Groq AI analysis ---
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        return _error(500, "GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=groq_key)

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
        return _error(500, "AI returned invalid JSON. Please try again.")
    except Exception as exc:
        return _error(500, f"AI analysis failed: {exc}")

    result["filename"] = filename
    result["analyzed_at"] = datetime.utcnow().isoformat()

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(result),
    }