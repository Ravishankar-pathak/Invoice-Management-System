import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import json
import psycopg2
import ollama
import re
import os
import logging
from datetime import datetime
import numpy as np
import cv2
import io
import time
import subprocess
import pytz

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("invoice_parser.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Function to get current date and time in IST
def get_current_datetime():
    ist = pytz.timezone('Asia/Kolkata')  # Adjust timezone as needed
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

# Function to preprocess the image for OCR with advanced techniques
def preprocess_image(image):
    img_np = np.array(image)
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=1)
    enhanced_image = Image.fromarray(dilated)
    enhancer = ImageEnhance.Contrast(enhanced_image)
    enhanced_image = enhancer.enhance(2)
    enhanced_image = enhanced_image.filter(ImageFilter.SHARPEN)
    return enhanced_image

# Function to extract text from PDF using pdfplumber and advanced OCR
def extract_text_from_pdf(pdf_path):
    text_data = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    text = improve_formatting(text)
                    logger.info(f"Extracted text from page {i} (pdfplumber)")
                    text_data.append(text)
                else:
                    logger.info(f"Direct text extraction failed for page {i}, trying OCR")
                    page_image = page.to_image(resolution=300)
                    pil_image = page_image.original
                    processed_image = preprocess_image(pil_image)
                    ocr_text = pytesseract.image_to_string(processed_image, lang='eng', config='--psm 6')
                    if len(ocr_text.strip()) < 50:
                        logger.info(f"OCR with PSM 6 failed, trying with PSM 3")
                        ocr_text = pytesseract.image_to_string(processed_image, lang='eng', config='--psm 3')
                    if ocr_text.strip():
                        ocr_text = improve_formatting(ocr_text)
                        logger.info(f"Extracted text from page {i} (OCR)")
                        text_data.append(ocr_text)
                    else:
                        logger.warning(f"Failed to extract text from page {i}")
        return text_data
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return []

# Function to improve the formatting of the extracted text
def improve_formatting(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]{2,})([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"l(?=\d)", "1", text)
    text = re.sub(r"O(?=\d)", "0", text)
    text = re.sub(r"(\d)l(\d)", r"\1/\2", text)
    text = re.sub(r"(\d),(\d{3})\.(\d{2})", r"\1\2.\3", text)
    text = re.sub(r"(\d{2})[oO]([A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9])", r"\1O\2", text)
    return text

# Function to test Ollama connection with retry
def test_ollama_connection(max_retries=3):
    for attempt in range(max_retries):
        try:
            logger.info(f"Testing Ollama connection... (Attempt {attempt+1}/{max_retries})")
            test_response = ollama.generate(
                model="mistral",
                prompt="Return 'Connection successful!'",
                options={"num_predict": 50}
            )
            response = test_response['response'].strip()
            logger.info(f"Test response: {response}")
            return True
        except Exception as e:
            logger.error(f"Ollama connection failed (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                logger.info("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                logger.error("Max retries reached. Falling back to default JSON.")
                return False

# Function to perform key-value extraction from text
def extract_key_value_pairs(text):
    patterns = {
        "invoice_number": [
            r"(\d{7,}|(?:[A-Za-z0-9\-/]+))\s+(\d{1,2}-[A-Za-z]{3}-\d{2,4})",  # Matches "2024164 7-Dec-24"
            r"Invoice\s*No\.\s*(\d{7,})",
            r"Trip\s*ID\s*([A-Za-z0-9\-]+)",
            r"Invoice\s*Nr\.\s*([A-Za-z0-9\-/]+)",
            r"Invoice\s*No\.?:?\s*[#]?([A-Za-z0-9\-/]+)",
            r"POO/\s*CN\s*No\.\s*([A-Za-z0-9\-]+)"
        ],
        "invoice_date": [
            r"(\d{7,}|(?:[A-Za-z0-9\-/]+))\s+(\d{1,2}-[A-Za-z]{3}-\d{2,4})",  # Matches "2024164 7-Dec-24"
            r"Dated\s*(\d{1,2}-[A-Za-z]{3}-\d{2,4})",
            r"Trip\s*Start\s*Date\s*(\d{2}/\d{2}/\d{4})",
            r"Date\s*&\s*Time\s*(\d{2}/\d{2}/\d{4})",
            r"Invoice\s*Date:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
            r"(\w+\s*\d{1,2},\s*\d{4})"
        ],
        "gst_number": [
            r"GSTIN/UIN:\s*([0-9A-Z]{15})",
            r"GSTIN:\s*([0-9A-Z]{15})"
        ],
        "company_name": [
            r"(TRENDING\s*TESTING\s*SERVICES\s*AND\s*LABS)",
            r"^(WheelsEye\s*Logistics\s*Private\s*Limited)",
            r"Sold\s*By\s*:\s*([A-Za-z0-9\s&\.,]+)(?=\n)",
            r"(Krishna\s*Smart\s*Technology)",
            r"(CONNECTORDEVICES)"
        ],
        "total_amount": [
            r"Total\s*Amount\s*₹?([\d,]+\.\d{2})",
            r"Grand\s*Total:?\s*(?:Rs\.?|₹|INR)?\s*([\d,]+\.\d{2})",
            r"Amount\s*Chargeable\s*(?:Rs\.?|₹|INR)?\s*([\d,]+\.\d{2})",
            r"Total\s*(\d+)\s*QTY\s*=\s*([\d,]+\.\d{2})"
        ]
    }
    
    extracted_data = {}
    for field, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                if field == "invoice_number" and pattern == r"(\d{7,}|(?:[A-Za-z0-9\-/]+))\s+(\d{1,2}-[A-Za-z]{3}-\d{2,4})":
                    value = matches.group(1).strip()
                elif field == "invoice_date" and pattern == r"(\d{7,}|(?:[A-Za-z0-9\-/]+))\s+(\d{1,2}-[A-Za-z]{3}-\d{2,4})":
                    value = matches.group(2).strip()
                elif field == "total_amount" and len(matches.groups()) > 1:
                    value = matches.group(2).strip()
                else:
                    value = matches.group(1).strip()
                if field == "total_amount":
                    value = value.replace(",", "")
                if field == "gst_number":
                    value = value.replace("O", "0").upper()
                extracted_data[field] = value
                break
    return extracted_data

# Function to extract items from tabular data
def extract_items(text):
    items = []
    # Pattern to match service items (inspired by old code's Mistral output)
    item_pattern = r"([\w\s\-&]+?)\s+(\d{4,8})?\s*(\d+(?:\s*nos)?)\s*(NOS|QTY|nos)?\s*([\d,]+\.\d{2})\s*([\d,]+\.\d{2})?"
    item_matches = re.finditer(item_pattern, text, re.IGNORECASE)
    
    for i, match in enumerate(item_matches, 1):
        description = match.group(1).strip()
        hsn = match.group(2) if match.group(2) else "N/A"
        quantity = match.group(3).replace(" nos", "").strip()
        unit = match.group(4).upper() if match.group(4) else "QTY"
        rate = match.group(5).replace(",", "") if match.group(5) else "0.00"
        amount = match.group(6).replace(",", "") if match.group(6) else rate
        
        # Extract GST percentage from text
        gst_percent_pattern = r"(\d+)%"
        gst_percent_match = re.search(gst_percent_pattern, text, re.IGNORECASE)
        gst_percent = gst_percent_match.group(1) + "%" if gst_percent_match else "N/A"
        
        item = {
            "sr_no": str(i),
            "description": description,
            "hsn": hsn,
            "gst": gst_percent,
            "quantity": quantity,
            "unit": unit,
            "rate": rate,
            "amount": amount
        }
        items.append(item)
    
    # Alternative pattern for cases where HSN or amount is missing
    if not items:
        alt_pattern = r"([\w\s\-&]+?)\s*(\d+(?:\s*nos)?)\s*(NOS|QTY|nos)?\s*([\d,]+\.\d{2})"
        alt_matches = re.finditer(alt_pattern, text, re.IGNORECASE)
        for i, match in enumerate(alt_matches, 1):
            description = match.group(1).strip()
            quantity = match.group(2).replace(" nos", "").strip()
            unit = match.group(3).upper() if match.group(3) else "QTY"
            amount = match.group(4).replace(",", "")
            item = {
                "sr_no": str(i),
                "description": description,
                "hsn": "N/A",
                "gst": "N/A",
                "quantity": quantity,
                "unit": unit,
                "rate": amount,  # Assume rate = amount if not specified
                "amount": amount
            }
            items.append(item)
    
    return items

# Function for analyzing with Mistral
def analyze_with_mistral(text):
    if not test_ollama_connection():
        logger.warning("Mistral server unavailable, falling back to regex extraction")
        extracted_kv = extract_key_value_pairs(text)
        extracted_items = extract_items(text)
        default_structure = {
            "invoice_number": extracted_kv.get("invoice_number", "UNKNOWN"),
            "invoice_date": extracted_kv.get("invoice_date", "N/A"),
            "customer_name": "Eigen Technologies Pvt Ltd",
            "gst_number": extracted_kv.get("gst_number", "UNKNOWN"),
            "company_name": extracted_kv.get("company_name", "UNKNOWN"),
            "total_amount": extracted_kv.get("total_amount", "0.00"),
            "items": extracted_items
        }
        return json.dumps(default_structure, ensure_ascii=False)

    try:
        logger.info("Starting Mistral processing...")
        prompt = (
            "Extract the following details from the invoice text and return them in a valid JSON string:\n"
            "{\n"
            "  \"invoice_number\": \"<number>\",\n"
            "  \"invoice_date\": \"<date>\",\n"
            "  \"customer_name\": \"Eigen Technologies Pvt Ltd\",\n"
            "  \"gst_number\": \"<only the GST number of the company sending the invoice>\",\n"
            "  \"company_name\": \"<company sending the invoice, NOT Eigen Technologies Pvt Ltd>\",\n"
            "  \"total_amount\": \"<amount with commas removed, e.g., 30680.00>\",\n"
            "  \"items\": [\n"
            "    {\"sr_no\": \"1\", \"description\": \"<desc>\", \"hsn\": \"<hsn>\", \"gst\": \"<gst_rate>\", \"quantity\": \"<qty>\", \"unit\": \"<unit, default to QTY if not specified>\", \"rate\": \"<rate, commas removed>\", \"amount\": \"<amount, commas removed>\"}\n"
            "    (and so on for each item, incrementing sr_no). DO NOT add empty objects or placeholders if no more items exist.\n"
            "  ]\n"
            "}\n"
            "Rules to follow ALWAYS:\n"
            "1. STRICT RULE: If invoice number and date appear together in the format like '2024164 7-Dec-24', 'SBA/24-25/1782 2-Dec-24', or 'MEPL/23-24/003 17-Apr-23', ALWAYS assign the number or code (e.g., '2024164', 'SBA/24-25/1782', 'MEPL/23-24/003') as invoice_number and the date (e.g., '7-Dec-24', '2-Dec-24', '17-Apr-23') as invoice_date. This rule takes precedence over all others for invoice_number and invoice_date.\n"
            "2. If written as 'Invoice number = 12137sb3' and 'Invoice date = 2/3/2020', preserve them as is.\n"
            "3. Extract only the sender company's GST number (e.g., '09BKDPJ6675A1Z6').\n"
            "4. Remove commas from total_amount, rate, and amount fields (e.g., 30680.00, not 30,680.00).\n"
            "5. Use 'QTY' as unit if not specified in the invoice.\n"
            "6. Include all items with incrementing sr_no starting from 1.\n"
            "7. Return only the JSON string, no extra text outside the JSON.\n"
            "8. If written as 'Tax Invoice No : JP/2324/0009 Date : 5-Apr-23', then 'JP/2324/0009' is invoice_number and '5-Apr-23' is invoice_date.\n"
            "9. STRICT RULE: For 'total_amount', extract the value EXACTLY as per the invoice text from lines like 'Total Amount', 'Grand Total', or 'Amount Chargeable' (e.g., '30680.00'). Do NOT recalculate it.\n"
            "10. 'customer_name' MUST ALWAYS be 'Eigen Technologies Pvt Ltd', and 'company_name' is the sender (e.g., 'Trending Testing Services and Labs').\n"
            "11. 'items' list includes ONLY the services listed under 'Description of Services' (e.g., 'Temp Cycle Test', 'Type Test', 'Surge Test'). Do NOT include 'IGST', 'CGST', 'Freight', 'ROUNDOFF', or any tax lines as they don't have 'hsn', 'gst', 'rate'.\n"
            "12. Extract EXACT quantity, rate, and amount from each item row. If '1 nos' or '2 nos', use '1' or '2' as quantity and 'NOS' as unit.\n"
            "13. Total Amount, Grand Total, or the highest amount in the entire invoice must be recorded as the Total Amount — this is a strict rule for total amount extraction.\n"
            "Ignore all other details and ensure the output is a parseable JSON object.\n"
            f"\n\n{text}"
        )
        response = ollama.generate(
            model="mistral",
            prompt=prompt,
            options={
                "temperature": 0.5,
                "top_p": 0.9,
                "num_predict": 3000  # Increased to handle multiple items
            }
        )
        mistral_output = response['response'].strip()
        logger.info("Mistral processing complete")
        print(f"\n=== Mistral Output ===\n{mistral_output}\n=== End of Mistral Output ===\n")
        if not mistral_output:
            raise ValueError("Mistral returned empty output")
        try:
            json.loads(mistral_output)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from Mistral, falling back to regex extraction")
            return analyze_with_mistral_fallback(text)
        return mistral_output
    except Exception as e:
        logger.error(f"Error with Mistral generation: {e}")
        return analyze_with_mistral_fallback(text)

# Fallback method using regex extraction
def analyze_with_mistral_fallback(text):
    logger.info("Using fallback extraction method")
    extracted_kv = extract_key_value_pairs(text)
    extracted_items = extract_items(text)
    default_structure = {
        "invoice_number": extracted_kv.get("invoice_number", "UNKNOWN"),
        "invoice_date": extracted_kv.get("invoice_date", "N/A"),
        "customer_name": "Eigen Technologies Pvt Ltd",
        "gst_number": extracted_kv.get("gst_number", "UNKNOWN"),
        "company_name": extracted_kv.get("company_name", "UNKNOWN"),
        "total_amount": extracted_kv.get("total_amount", "0.00"),
        "items": extracted_items
    }
    return json.dumps(default_structure, ensure_ascii=False)

# Function to parse and fix JSON output
def extract_invoice_fields(text):
    default_structure = {
        "invoice_number": "UNKNOWN",
        "invoice_date": "N/A",
        "customer_name": "Eigen Technologies Pvt Ltd",
        "gst_number": "UNKNOWN",
        "company_name": "UNKNOWN",
        "total_amount": "0.00",
        "items": []
    }
    try:
        structured_data = json.loads(text)
        logger.info("Successfully parsed JSON output")
        
        structured_data["customer_name"] = "Eigen Technologies Pvt Ltd"
        
        for key in default_structure:
            if key not in structured_data or structured_data[key] is None or structured_data[key] == "":
                structured_data[key] = default_structure[key]
        
        if "total_amount" in structured_data and isinstance(structured_data["total_amount"], str):
            structured_data["total_amount"] = structured_data["total_amount"].replace(",", "")
        
        if "items" in structured_data and isinstance(structured_data["items"], list):
            fixed_items = []
            for i, item in enumerate(structured_data["items"], 1):
                if not isinstance(item, dict):
                    continue
                default_item = {
                    "sr_no": str(i),
                    "description": "N/A",
                    "hsn": "N/A",
                    "gst": "N/A",
                    "quantity": "0",
                    "unit": "NOS",
                    "rate": "0.00",
                    "amount": "0.00"
                }
                fixed_item = dict(default_item)
                for key, value in item.items():
                    if key in fixed_item:
                        if key in ["rate", "amount"] and isinstance(value, str):
                            fixed_item[key] = value.replace(",", "")
                        else:
                            fixed_item[key] = str(value)
                fixed_items.append(fixed_item)
            structured_data["items"] = fixed_items
        else:
            structured_data["items"] = []
        
        keys_to_keep = default_structure.keys()
        structured_data = {k: v for k, v in structured_data.items() if k in keys_to_keep}
        
        return structured_data
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.info("Attempting to extract data using regex patterns")
        extracted_kv = extract_key_value_pairs(text)
        extracted_items = extract_items(text)
        default_structure.update({
            "invoice_number": extracted_kv.get("invoice_number", "UNKNOWN"),
            "invoice_date": extracted_kv.get("invoice_date", "N/A"),
            "customer_name": "Eigen Technologies Pvt Ltd",
            "gst_number": extracted_kv.get("gst_number", "UNKNOWN"),
            "company_name": extracted_kv.get("company_name", "UNKNOWN"),
            "total_amount": extracted_kv.get("total_amount", "0.00"),
            "items": extracted_items
        })
        return default_structure

# Function to validate invoice data (no total amount recalculation)
def validate_invoice_data(structured_data):
    try:
        if not structured_data.get("items", []):
            return structured_data
            
        if "gst_number" in structured_data and structured_data["gst_number"]:
            structured_data["gst_number"] = structured_data["gst_number"].replace("O", "0").upper()
        
        if "total_amount" in structured_data and isinstance(structured_data["total_amount"], str):
            structured_data["total_amount"] = structured_data["total_amount"].replace(",", "")
        
        return structured_data
    except Exception as e:
        logger.error(f"Error in validation: {e}")
        return structured_data

# Function to save data into PostgreSQL
def save_data_to_postgresql(data):
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="your database name",
            user="your username ",
            password="your password"
        )
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Datas_of_invoice (
                id SERIAL PRIMARY KEY,
                invoice_number VARCHAR(50),
                invoice_date VARCHAR(20),
                customer_name VARCHAR(100),
                gst_number VARCHAR(15),
                company_name VARCHAR(100),
                total_amount DECIMAL(15,2),
                items JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        invoice_date_raw = data['invoice_date'] if data['invoice_date'] != "N/A" else None
        if invoice_date_raw:
            try:
                date_formats = [
                    "%d/%m/%Y",
                    "%d-%b-%y",
                    "%Y-%m-%d",
                    "%d-%m-%Y",
                    "%B %d, %Y"
                ]
                converted = False
                for fmt in date_formats:
                    try:
                        invoice_date_obj = datetime.strptime(invoice_date_raw, fmt)
                        invoice_date = invoice_date_obj.strftime("%Y-%m-%d")
                        converted = True
                        break
                    except ValueError:
                        continue
                if not converted:
                    logger.warning(f"Could not parse date '{invoice_date_raw}' with any known format")
                    invoice_date = None
            except Exception as e:
                logger.warning(f"Invalid date format '{invoice_date_raw}', saving as NULL: {e}")
                invoice_date = None
        else:
            invoice_date = None
        
        try:
            total_amount = float(data['total_amount']) if data['total_amount'] else 0.00
        except (ValueError, TypeError):
            logger.warning(f"Invalid total_amount '{data['total_amount']}', defaulting to 0.00")
            total_amount = 0.00
        
        insert_query = """
        INSERT INTO Datas_of_invoice (invoice_number, invoice_date, customer_name, gst_number, company_name, total_amount, items)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        items_json = json.dumps(data['items'])
        cursor.execute(insert_query, (
            data['invoice_number'],
            invoice_date,
            data['customer_name'],
            data['gst_number'],
            data['company_name'],
            total_amount,
            items_json
        ))
        conn.commit()
        logger.info(f"Data successfully saved in PostgreSQL for invoice {data['invoice_number']}")
    except psycopg2.Error as e:
        logger.error(f"PostgreSQL error: {e.pgcode} - {e.pgerror}")
        if conn:
            conn.rollback()
    except Exception as e:
        logger.error(f"Error saving data to PostgreSQL: {e}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Comprehensive database of electronic component types and their applicable standards
ELECTRONIC_COMPONENT_STANDARDS = {
    "connector": {
        "standards": {
            "IEC": "IEC 60603-2",
            "RoHS": "Directive 2011/65/EU",
            "UL": "UL 1977",
            "IPC": "IPC-A-610"
        },
        "keywords": ["connector", "conn", "pin", "socket", "plug", "terminal", "header", "receptacle"],
        "description": "Connectors join electrical circuits together. IEC 60603-2 ensures connector safety and interoperability, RoHS restricts hazardous substances, UL 1977 certifies component safety, IPC-A-610 defines acceptability criteria."
    },
    "resistor": {
        "standards": {
            "IEC": "IEC 60115-1",
            "RoHS": "Directive 2011/65/EU",
            "EIA": "EIA-RS-279"
        },
        "keywords": ["resistor", "ohm", "k ohm", "kohm", "m ohm", "mohm"],
        "description": "Resistors limit current flow in electronic circuits. IEC 60115-1 defines performance requirements, RoHS ensures material compliance, EIA-RS-279 provides resistor color coding."
    },
    "capacitor": {
        "standards": {
            "IEC": "IEC 60384-1",
            "RoHS": "Directive 2011/65/EU",
            "EIA": "EIA-198"
        },
        "keywords": ["capacitor", "cap", "uf", "μf", "pf", "nf", "mf", "farad", "ceramic", "tantalum", "electrolytic"],
        "description": "Capacitors store electrical energy in an electric field. IEC 60384-1 ensures reliability, RoHS restricts hazardous materials, EIA-198 provides capacitor marking standards."
    },
    "diode": {
        "standards": {
            "IEC": "IEC 60747-7",
            "RoHS": "Directive 2011/65/EU",
            "JEDEC": "JESD282-B"
        },
        "keywords": ["diode", "led", "rectifier", "zener", "schottky", "tvs"],
        "description": "Diodes allow current flow in one direction. IEC 60747-7 defines semiconductor device specifications, RoHS ensures material compliance, JEDEC JESD282-B provides diode standards."
    },
    "ic": {
        "standards": {
            "IEC": "IEC 60748-1",
            "RoHS": "Directive 2011/65/EU",
            "JEDEC": "JESD47"
        },
        "keywords": ["ic", "integrated circuit", "microcontroller", "processor", "mcu", "microchip", "chip", "soc"],
        "description": "Integrated circuits combine multiple electronic components on a semiconductor substrate. IEC 60748-1 defines reliability standards, RoHS ensures material compliance, JEDEC JESD47 covers stress-test qualification."
    },
    "inductor": {
        "standards": {
            "IEC": "IEC 60938-1",
            "RoHS": "Directive 2011/65/EU"
        },
        "keywords": ["inductor", "coil", "choke", "transformer", "henry", "uh", "mh", "h"],
        "description": "Inductors store energy in a magnetic field. IEC 60938-1 defines performance standards, RoHS ensures material compliance."
    },
    "transistor": {
        "standards": {
            "IEC": "IEC 60747-8",
            "RoHS": "Directive 2011/65/EU",
            "JEDEC": "JESD282-B"
        },
        "keywords": ["transistor", "bjt", "fet", "mosfet", "jfet", "igbt"],
        "description": "Transistors amplify or switch electronic signals. IEC 60747-8 defines transistor specifications, RoHS ensures material compliance, JEDEC JESD282-B provides standards."
    },
    "pcb": {
        "standards": {
            "IPC": "IPC-6012",
            "RoHS": "Directive 2011/65/EU",
            "UL": "UL 796"
        },
        "keywords": ["pcb", "printed circuit board", "circuit board", "board", "substrate"],
        "description": "PCBs mechanically support and electrically connect components. IPC-6012 defines qualification requirements, RoHS ensures material compliance, UL 796 certifies PCB safety."
    },
    "switch": {
        "standards": {
            "IEC": "IEC 61058-1",
            "RoHS": "Directive 2011/65/EU",
            "UL": "UL 1054"
        },
        "keywords": ["switch", "toggle", "push button", "pushbutton", "tactile", "dip switch"],
        "description": "Switches interrupt or divert current flow. IEC 61058-1 establishes safety requirements, RoHS ensures material compliance, UL 1054 certifies switch safety."
    },
    "relay": {
        "standards": {
            "IEC": "IEC 61810-1",
            "RoHS": "Directive 2011/65/EU",
            "UL": "UL 508"
        },
        "keywords": ["relay", "contactor"],
        "description": "Relays control circuits with low-power signals. IEC 61810-1 defines electromechanical requirements, RoHS ensures material compliance, UL 508 certifies industrial control equipment."
    },
    "crystal": {
        "standards": {
            "IEC": "IEC 60122-1",
            "RoHS": "Directive 2011/65/EU"
        },
        "keywords": ["crystal", "oscillator", "resonator", "xtal", "mhz", "khz"],
        "description": "Crystals provide precise frequency control. IEC 60122-1 defines quartz crystal specifications, RoHS ensures material compliance."
    },
    "fuse": {
        "standards": {
            "IEC": "IEC 60127",
            "RoHS": "Directive 2011/65/EU",
            "UL": "UL 248"
        },
        "keywords": ["fuse", "circuit protection", "breaker"],
        "description": "Fuses protect circuits from overcurrent. IEC 60127 defines miniature fuse safety requirements, RoHS ensures material compliance, UL 248 certifies fuses."
    },
    "sensor": {
        "standards": {
            "IEC": "IEC 61000-4-2",
            "RoHS": "Directive 2011/65/EU",
            "ISO": "ISO 9001"
        },
        "keywords": ["sensor", "transducer", "detector"],
        "description": "Sensors detect physical properties and convert them to signals. IEC 61000-4-2 covers EMC requirements, RoHS ensures material compliance, ISO 9001 ensures quality management."
    },
    "battery": {
        "standards": {
            "IEC": "IEC 60086",
            "RoHS": "Directive 2006/66/EC",
            "UL": "UL 1642"
        },
        "keywords": ["battery", "cell", "accumulator", "lithium", "li-ion", "nimh", "nicd"],
        "description": "Batteries store energy chemically. IEC 60086 defines primary battery specifications, EU Directive 2006/66/EC covers battery regulations, UL 1642 certifies lithium batteries."
    },
    "led": {
        "standards": {
            "IEC": "IEC 62031",
            "RoHS": "Directive 2011/65/EU",
            "UL": "UL 8750"
        },
        "keywords": ["led", "light emitting diode", "oled", "indicator"],
        "description": "LEDs emit light when current flows. IEC 62031 provides safety specifications, RoHS ensures material compliance, UL 8750 certifies LED equipment."
    },
    "module": {
        "standards": {
            "IEC": "IEC 61131-2",
            "RoHS": "Directive 2011/65/EU",
            "ISO": "ISO 9001"
        },
        "keywords": ["module", "assembly", "converter", "regulator", "driver"],
        "description": "Electronic modules are functional units containing multiple components. IEC 61131-2 defines equipment requirements, RoHS ensures material compliance, ISO 9001 ensures quality."
    }
}

# Default standards to use if component type cannot be determined
DEFAULT_COMPONENT_STANDARDS = {
    "standards": {
        "RoHS": "Directive 2011/65/EU",
        "ISO": "ISO 9001"
    },
    "description": "Electronic component with RoHS compliance for hazardous substance restrictions and manufactured under ISO 9001 quality management standards."
}

def identify_component_type(component_name):
    if not component_name or not isinstance(component_name, str):
        return "unknown", 0.0
    
    comp_name_lower = component_name.lower()
    
    for comp_type, data in ELECTRONIC_COMPONENT_STANDARDS.items():
        if comp_type.lower() in comp_name_lower:
            return comp_type, 0.9
    
    best_match = "unknown"
    best_score = 0.0
    
    for comp_type, data in ELECTRONIC_COMPONENT_STANDARDS.items():
        keywords = data.get("keywords", [])
        
        for keyword in keywords:
            if keyword.lower() in comp_name_lower:
                confidence = min(0.85, (len(keyword) / len(comp_name_lower)) * 0.85)
                
                if confidence > best_score:
                    best_match = comp_type
                    best_score = confidence
    
    if best_score < 0.1:
        if re.search(r'[A-Za-z]+\d+', comp_name_lower) or re.search(r'\d+[A-Za-z]+', comp_name_lower):
            return "ic", 0.3
            
    return best_match, best_score

def get_reliable_standards(component_name, company_name, invoice_number):
    try:
        logger.info(f"Getting reliable standards for: {component_name}")
        
        comp_type, confidence = identify_component_type(component_name)
        
        logger.info(f"Identified '{component_name}' as '{comp_type}' with confidence {confidence:.2f}")
        
        if comp_type != "unknown" and confidence >= 0.1:
            standards_info = ELECTRONIC_COMPONENT_STANDARDS[comp_type]
            description = standards_info["description"]
            
            if confidence < 0.5:
                description = f"Based on its name, this component appears to be a {comp_type}. {description}"
        else:
            standards_info = DEFAULT_COMPONENT_STANDARDS
            description = f"Component type could not be determined with confidence for '{component_name}'. {standards_info['description']}"
        
        quality_info = {
            "component": component_name,
            "standards": standards_info["standards"],
            "description": description,
            "current_date_time": get_current_datetime(),
            "company_name": company_name,
            "invoice_number": invoice_number
        }
        
        return quality_info
        
    except Exception as e:
        logger.error(f"Error getting reliable standards for {component_name}: {e}")
        return {
            "component": component_name,
            "standards": {
                "RoHS": "Directive 2011/65/EU"
            },
            "description": f"Error occurred while determining standards for this component. Default RoHS compliance is assumed.",
            "current_date_time": get_current_datetime(),
            "company_name": company_name,
            "invoice_number": invoice_number
        }

def advanced_component_name_processing(component_name):
    if not component_name or not isinstance(component_name, str):
        return component_name
    
    processed = re.sub(r'\s+', ' ', component_name.strip()).lower()
    
    abbreviation_map = {
        r'\bres\.?\b': 'resistor',
        r'\bcap\.?\b': 'capacitor',
        r'\bconn\.?\b': 'connector',
        r'\btrans\.?\b': 'transistor',
        r'\bxtal\.?\b': 'crystal',
        r'\bic\.?\b': 'integrated circuit',
        r'\bpcb\.?\b': 'printed circuit board',
        r'\bred\s+led\b': 'led red',
        r'\bgreen\s+led\b': 'led green',
        r'\bblue\s+led\b': 'led blue'
    }
    
    misspelling_map = {
        r'\bresistor\b': 'resistor',
        r'\bressistor\b': 'resistor',
        r'\bcapaciter\b': 'capacitor',
        r'\bcapasitor\b': 'capacitor',
        r'\bconector\b': 'connector',
        r'\bconnecter\b': 'connector'
    }
    
    for pattern, replacement in abbreviation_map.items():
        processed = re.sub(pattern, replacement, processed)
    
    for pattern, replacement in misspelling_map.items():
        processed = re.sub(pattern, replacement, processed)
    
    return processed

def process_quality_standards_for_invoice(structured_data):
    quality_data = []
    
    try:
        total_items = len(structured_data.get("items", []))
        logger.info(f"Processing quality standards for {total_items} items in invoice {structured_data.get('invoice_number', 'unknown')}")
        
        for item in structured_data.get("items", []):
            component_name = item.get("description", "").strip()
            
            if component_name and component_name != "N/A":
                try:
                    processed_name = advanced_component_name_processing(component_name)
                    
                    quality_info = get_reliable_standards(
                        processed_name,
                        structured_data.get("company_name", "Unknown Company"),
                        structured_data.get("invoice_number", "Unknown Invoice")
                    )
                    
                    quality_data.append(quality_info)
                    logger.info(f"Successfully derived standards for item: {component_name}")
                    
                except Exception as e:
                    logger.error(f"Error processing quality standards for {component_name}: {e}")
                    fallback_info = {
                        "component": component_name,
                        "standards": {"RoHS": "Directive 2011/65/EU"},
                        "description": f"Electronic component with minimum RoHS compliance. Standards extraction failed due to an error.",
                        "current_date_time": get_current_datetime(),
                        "company_name": structured_data.get("company_name", "Unknown Company"),
                        "invoice_number": structured_data.get("invoice_number", "Unknown Invoice")
                    }
                    quality_data.append(fallback_info)
            else:
                logger.warning(f"Skipping quality standards for empty component description in item {item.get('sr_no', 'unknown')}")
    
    except Exception as e:
        logger.error(f"Error in quality standards processing: {e}")
    
    return quality_data

# Define IQC questions
IQC_QUESTIONS = [
    "Have all incoming components been physically inspected for damage or defects? If Yes, no issues found; if No, what issues were found?",
    "Do the components match the specifications listed in the invoice? If No, what mismatches were found?",
    "Are the required quality standards (e.g., IEC, RoHS) met for all components? If No, which standards are not met?",
    "Is the quantity received consistent with the invoice? If No, what is the actual quantity received?",
    "Are there any discrepancies in the packaging or labeling? If Yes, what discrepancies were found?",
    "Is the color of the components as expected? Please specify the expected and observed color.",
    "Have samples been tested for functionality (if applicable)? If No, why not?",
    "Is the supplier's documentation (e.g., certificate of conformity) provided and valid? If No, what is missing or invalid?"
]

# Function to perform IQC Q&A with the user
def perform_iqc_qa(invoice_number):
    iqc_responses = {}
    print("\nStarting Incoming Quality Control (IQC) Verification...")
    print(f"Invoice Number: {invoice_number}")
    print("Please answer the following questions:\n")
    
    for i, question in enumerate(IQC_QUESTIONS, 1):
        while True:
            response = input(f"Q{i}: {question} (Yes/No) ").strip().lower()
            if response in ["yes", "no", "y", "n"]:
                answer = "Yes" if response in ["yes", "y"] else "No"
                details = ""
                if answer == "No" or (i == 6 and answer == "Yes"):
                    if i == 1:
                        details = input("What issues were found? (Enter 'None' if no details): ") if answer == "No" else "No issues found"
                    elif i == 2:
                        details = input("What mismatches were found? (Enter 'None' if no details): ") if answer == "No" else "Matches invoice"
                    elif i == 3:
                        details = input("Which standards are not met? (Enter 'None' if no details): ") if answer == "No" else "All standards met"
                    elif i == 4:
                        details = input("What is the actual quantity received? (Enter 'None' if no details): ") if answer == "No" else "Matches invoice"
                    elif i == 5:
                        details = input("What discrepancies were found? (Enter 'None' if no details): ") if answer == "Yes" else "No discrepancies"
                    elif i == 6:
                        expected = input("What was the expected color? ")
                        observed = input("What was the observed color? ")
                        details = f"Expected: {expected}, Observed: {observed}"
                    elif i == 7:
                        details = input("Why not? (Enter 'None' if no details): ") if answer == "No" else "Tested successfully"
                    elif i == 8:
                        details = input("What is missing or invalid? (Enter 'None' if no details): ") if answer == "No" else "All documentation valid"
                iqc_responses[f"Q{i}"] = {
                    "question": question,
                    "answer": answer,
                    "details": details if details else "None"
                }
                break
            else:
                print("please answer with 'Yes' or 'No' (or 'y'/'n').")
    
    iqc_status = "PASS" if all(q["answer"] == "Yes" for q in iqc_responses.values()) else "FAIL"
    iqc_responses["iqc_status"] = iqc_status
    
    logger.info(f"IQC completed for invoice {invoice_number}. Status: {iqc_status}")
    return iqc_responses

# Function to validate and correct JSON data post-IQC
def validate_and_correct_json(combined_data):
    print("\nReview the extracted data:")
    print(json.dumps(combined_data, indent=2))
    print("\nNow, let's verify each field.\n")

    invoice_data = combined_data["invoice_data"]
    for field in invoice_data.keys():
        if field == "items":
            for i, item in enumerate(invoice_data["items"], 1):
                print(f"\nItem {i}:")
                for item_field in item.keys():
                    while True:
                        response = input(f"Is '{item_field}' ({item[item_field]}) correct for Item {i}? (Yes/No) ").strip().lower()
                        if response in ["yes", "y"]:
                            break
                        elif response in ["no", "n"]:
                            corrected_value = input(f"Enter the correct value for '{item_field}': ")
                            item[item_field] = corrected_value
                            break
                        else:
                            print("Please answer with 'Yes' or 'No' (or 'y'/'n').")
        else:
            while True:
                response = input(f"Is '{field}' ({invoice_data[field]}) correct? (Yes/No) ").strip().lower()
                if response in ["yes", "y"]:
                    break
                elif response in ["no", "n"]:
                    corrected_value = input(f"Enter the correct value for '{field}': ")
                    invoice_data[field] = corrected_value
                    break
                else:
                    print("Please answer with 'Yes' or 'No' (or 'y'/'n').")

    quality_standards = combined_data["quality_standards"]
    for i, standard in enumerate(quality_standards, 1):
        print(f"\nQuality Standard {i}:")
        for field in ["company_name", "current_date_time", "invoice_number"]:
            while True:
                response = input(f"Is '{field}' ({standard[field]}) correct for Quality Standard {i}? (Yes/No) ").strip().lower()
                if response in ["yes", "y"]:
                    break
                elif response in ["no", "n"]:
                    corrected_value = input(f"Enter the correct value for '{field}': ")
                    standard[field] = corrected_value
                    break
                else:
                    print("Please answer with 'Yes' or 'No' (or 'y'/'n').")

    iqc_responses = combined_data["iqc_responses"]
    for key, value in iqc_responses.items():
        if key != "iqc_status":
            while True:
                response = input(f"Is '{value['question']}' answered as '{value['answer']}' with details '{value['details']}' correct? (Yes/No) ").strip().lower()
                if response in ["yes", "y"]:
                    break
                elif response in ["no", "n"]:
                    corrected_answer = input("Enter the correct answer (Yes/No): ").strip().lower()
                    if corrected_answer in ["yes", "y"]:
                        value["answer"] = "Yes"
                        if key == "Q6":
                            expected = input("What was the expected color? ")
                            observed = input("What was the observed color? ")
                            value["details"] = f"Expected: {expected}, Observed: {observed}"
                        else:
                            value["details"] = "None" if key in ["Q1", "Q5"] else input(f"Enter the correct details for '{value['question']}': ")
                    elif corrected_answer in ["no", "n"]:
                        value["answer"] = "No"
                        value["details"] = input(f"Enter the correct details for '{value['question']}': ")
                    else:
                        print("Please enter 'Yes' or 'No' (or 'y'/'n').")
                        continue
                    break
                else:
                    print("Please answer with 'Yes' or 'No' (or 'y'/'n').")
    iqc_responses["iqc_status"] = "PASS" if all(q["answer"] == "Yes" for q in iqc_responses.values() if isinstance(q, dict)) else "FAIL"

    return combined_data

# Process invoice function
def process_invoice(pdf_path):
    try:
        logger.info(f"Starting processing for {pdf_path}")
        
        # Extract text from PDF
        text_data = extract_text_from_pdf(pdf_path)
        if not text_data:
            logger.error("No text extracted from PDF")
            return
        
        full_text = "\n".join(text_data)
        
        # Analyze with Mistral or fallback
        mistral_output = analyze_with_mistral(full_text)
        
        # Parse the output into structured data
        structured_data = extract_invoice_fields(mistral_output)
        
        # Validate data (no total amount recalculation)
        structured_data = validate_invoice_data(structured_data)
        
        # Extract component names from items and get quality standards
        quality_data = process_quality_standards_for_invoice(structured_data)
        
        # Perform IQC Q&A with the user
        iqc_responses = perform_iqc_qa(structured_data["invoice_number"])
        
        # Combine invoice data, quality standards, and IQC responses
        combined_data = {
            "invoice_data": structured_data,
            "quality_standards": quality_data,
            "iqc_responses": iqc_responses
        }
        
        # Validate and correct the combined data
        combined_data = validate_and_correct_json(combined_data)
        
        # Save to database (only invoice data)
        save_data_to_postgresql(combined_data["invoice_data"])
        
        # Save combined data to JSON file
        output_json_path = pdf_path.replace(".pdf", "_processed.json")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved combined JSON output to {output_json_path}")
        
        # Print summary for user
        print(f"\nProcessing complete for {pdf_path}")
        print(f"IQC Status: {combined_data['iqc_responses']['iqc_status']}")
        print(f"Output saved to: {output_json_path}")
        
    except Exception as e:
        logger.error(f"Error processing invoice {pdf_path}: {e}")

# Example usage
if __name__ == "__main__":
    pdf_path = "07-12-24-Trending Testing Services-1.pdf"  # Update path as per your file
    if os.path.exists(pdf_path):
        process_invoice(pdf_path)
    else:
        logger.error(f"PDF file not found: {pdf_path}")
