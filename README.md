# Invoice Processor with Quality Control 📄

A Python-based invoice processing system that extracts data from PDF invoices, validates them against quality standards, and stores the results in a PostgreSQL database.

## 🔍 Features
- **PDF Text Extraction**: Uses `pdfplumber` and advanced OCR with `pytesseract`.
- **Invoice Data Extraction**: Leverages `ollama` (Mistral model) for structured JSON output.
- **Quality Standards**: Identifies electronic component types and applies relevant standards (e.g., IEC, RoHS).
- **Incoming Quality Control (IQC)**: Interactive QA to verify invoice data against physical components.
- **Database Integration**: Stores processed invoice data in PostgreSQL.

## 🧰 Requirements
- Python 3.8+
- PostgreSQL 12+
- Tesseract OCR installed
- `ollama` running locally with Mistral model

## 📦 Dependencies
Install via:
```bash
pip install -r requirements.txt
# Invoice-Management-System
