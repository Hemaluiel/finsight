# FinSight — Bank Statement Analyzer

A full-stack web app to analyze bank statements with AI.
**Stack:** Python + FastAPI (backend) · HTML/CSS/JS (frontend) · SQLite (database) · Claude AI


## Features

- **PDF parsing** — pdfplumber extracts text from digital PDFs (not scanned)
- **AI analysis** — Claude categorizes debits, finds top 5, writes insights
- **Dashboard** — doughnut chart, horizontal bar chart, metric cards, top 5 list


## Notes

- Scanned/image PDFs won't work — only text-based PDFs (standard bank exports)
