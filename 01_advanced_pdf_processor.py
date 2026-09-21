import os
import sys
import re
import json
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pymupdf  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image

# Ensure stdout uses UTF-8 encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Define paths
WORKSPACE_DIR = Path(__file__).parent.resolve()
RAW_DATASET_DIR = WORKSPACE_DIR / "sllip_raw_dataset"
PROCESSED_DIR = WORKSPACE_DIR / "processed_data"
FAMILY_LAW_TXT_DIR = PROCESSED_DIR / "family_law_clean_txt"
STATUTORY_ACTS_TXT_DIR = PROCESSED_DIR / "statutory_acts_clean_txt"
GROUND_TRUTH_FILE = PROCESSED_DIR / "ground_truth.json"
EXTRACTION_METADATA_FILE = PROCESSED_DIR / "extraction_metadata.json"

# Ensure output directories exist
FAMILY_LAW_TXT_DIR.mkdir(parents=True, exist_ok=True)
STATUTORY_ACTS_TXT_DIR.mkdir(parents=True, exist_ok=True)

# Expanded Trilingual Family Law Target Keywords
FAMILY_LAW_KEYWORDS_ENGLISH = [
    "divorce", "custody", "maintenance", "alimony", "matrimonial",
    "wife", "husband", "marriage", "nullity", "guardianship", "paternity", "spousal"
]

FAMILY_LAW_KEYWORDS_SINHALA = [
    "දික්කසාද", "නඩත්තු", "භාරකාරත්වය", "විවාහ", "බිරිඳ", "ස්වාමියා", "අඹුසැමි"
]

FAMILY_LAW_KEYWORDS_TAMIL = [
    "விவாகரத்து", "பராமரிப்பு", "பாதுகாவல்", "திருமணம்"
]

ALL_FAMILY_LAW_KEYWORDS = (
    FAMILY_LAW_KEYWORDS_ENGLISH +
    FAMILY_LAW_KEYWORDS_SINHALA +
    FAMILY_LAW_KEYWORDS_TAMIL
)

# Verdict Matching RegEx Patterns (Trilingual)
VERDICT_PATTERNS = [
    (r"\b(?:appeal|application)\s+(?:is\s+)?allowed\s+in\s+part\b", "Appeal Allowed in Part"),
    (r"\b(?:appeal|application)\s+(?:is\s+)?allowed\b", "Appeal Allowed"),
    (r"\b(?:appeal|application)\s+(?:is\s+)?dismissed\b", "Appeal Dismissed"),
    (r"\b(?:order|judgment)\s+(?:is\s+)?set\s+aside\b", "Order Set Aside"),
    (r"\bset\s+aside\b", "Order Set Aside"),
    (r"\b(?:judgment|order)\s+(?:is\s+)?affirmed\b", "Judgment Affirmed"),
    (r"\baffirmed\b", "Judgment Affirmed"),
    (r"\bvaried\b|\bmodified\b", "Appeal Allowed in Part"),
    (r"\b(?:application)\s+(?:is\s+)?rejected\b", "Application Dismissed"),
    (r"\brejected\b", "Application Dismissed"),
    (r"අභියාචනය\s+අනුමතයි", "Appeal Allowed"),
    (r"අභියාචනය\s+ත්‍රෝන්\s+දමයි|අභියාචනය\s+ප්‍රතික්ෂේපයි", "Appeal Dismissed"),
    (r"மேல்முறையீடு\s+அனுமதிக்கப்பட்டது", "Appeal Allowed"),
    (r"மேல்முறையீடு\s+தள்ளுபடி", "Appeal Dismissed"),
]

# Legacy Non-Unicode Sinhala (FM-Abhaya / DL-Manel / Kaputa) to Unicode Mapping Table
SINHALA_LEGACY_MAP = [
    # Compound / Multi-character glyphs (Ordered longest first)
    ("Ì±Ç¶Æ¶Ç®Îº", "ප්‍රදේශීය"),
    ("±È½¹ÆÍ¹½Æ", "සභා"),
    ("¹ÇÕ¯¿Æ", "සභා"),
    ("±´Æ´ºÆ", "සංශෝධන"),
    ("¥Ý¹Æ‰´¯Æ", "පනත"),
    ("¥¾¯Æ¯", "පනත"),
    ("çÞ¯¿Æ", "පනත"),
    ("¤ºÆ", "අංක"),
    ("¤®ˆ", "අංක"),
    ("ÁÎ¾¹Æ‰¸Æ¸¾Æ", "අච්චුගස්වන"),
    ("ºÇ½ÆÞÛÎ½", "ලද්දේ"),
    ("¥¾+ÆÎ¯±", "පාර්ලිමේන්තු"),
    ("±¸·Ç»", "සභාවේ"),
    ("Í±Ç±‹±", "සම්මත"),
    ("Ðˆ»¼±È", "කළ"),
    ("¹Ç¼Çàº", "නියෝගය"),
    ("¶È¯¶È", "දින"),
    ("£¶Æ¶Ç´Æ", "මුද්‍රණය"),
    ("±È¹Æ¹Õ¶Æ¶¹Æ¹´Æ´×", "කරන ලදී"),
    ("£¼±È¸ÇÆ", "මිල"),
    ("¤Îµ¹Æ¹ˆ", "තැපැල්"),
    ("£±Æ±È´¹Æ¹´Æ´×", "ගාස්තුව"),
    ("¶¹Ç½Æ", "දේශීය"),
    ("Ì±¾â", "ප්‍රමාණය"),
    ("Í¶±È»", "සම්මත"),
    ("–´Îº¹ÆÚ", "නියෝගය"),
    ("¤Îµ»Ç¿¼Ç¾Æ", "තැපැල්"),
    ("£¾Æ¾×", "ගාස්තුව"),

    # Single character glyph replacements
    ("¤", "අ"), ("Æ", "්"), ("º", "න"), ("ˆ", "ි"), ("‰", "ී"),
    ("®", "ර"), ("¶", "ද"), ("½", "ස"), ("¿", "ත"), ("¾", "ත"),
    ("´", "න"), ("ç", "ප"), ("Þ", "න"), ("Á", "අ"), ("Ì", "ප්‍ර"),
    ("Î", "ි"), ("Ò", "ො"), ("Õ", "ො"), ("×", "්"), ("Ø", "ු"),
    ("Ù", "ූ"), ("Ú", "්"), ("Û", "ේ"), ("Ü", "ැ"), ("Ý", "න"),
    ("ß", "ල"), ("à", "ග"), ("á", "ක"), ("â", "ම"), ("ã", "ච"),
    ("ä", "ජ"), ("å", "ට"), ("æ", "ඩ"), ("è", "බ"), ("é", "ම"),
    ("ê", "ය"), ("ë", "ර"), ("ì", "ල"), ("í", "ව"), ("î", "ශ"),
    ("ï", "ෂ"), ("ð", "ස"), ("ñ", "හ"), ("ò", "ළ"), ("ó", "ෆ"),
    ("ô", "ං"), ("õ", "ඃ")
]

# Set of corrupted symbols for ratio calculations
CORRUPTED_ANSI_SYMBOLS = set("¤Æºˆ‰®¶½¿¾´çÞÁÌÎÒÕ×ØÙÚÛÜÝßàáâãäåæèéêëìíîïðñòóôtõö÷øùúûüýþÿ")


def convert_legacy_sinhala_fonts(text: str) -> str:
    """Normalize legacy Sinhala font characters into clean Unicode Sinhala text."""
    if not text:
        return ""
    converted = text
    for legacy_str, unicode_str in SINHALA_LEGACY_MAP:
        converted = converted.replace(legacy_str, unicode_str)
    return converted


def calculate_garbage_ratio(text: str) -> float:
    """Calculate the ratio of corrupted legacy ANSI symbols to non-whitespace characters."""
    if not text:
        return 0.0
    non_space_chars = [c for c in text if not c.isspace()]
    if not non_space_chars:
        return 0.0
    corrupted_count = sum(1 for c in non_space_chars if c in CORRUPTED_ANSI_SYMBOLS)
    return corrupted_count / len(non_space_chars)


# Check Tesseract OCR binary availability once
IS_TESSERACT_AVAILABLE = False
try:
    _langs = pytesseract.get_languages(config="")
    IS_TESSERACT_AVAILABLE = True
except Exception:
    IS_TESSERACT_AVAILABLE = False


def extract_text_from_pdf(filepath: Path) -> tuple[str, str]:
    """
    Layout-aware PDF extraction with legacy font conversion and OCR fallback.
    Returns: (text, extraction_method)
    """
    text_content = []
    extraction_method = "pymupdf_layout"

    try:
        doc = pymupdf.open(filepath)
        for page in doc:
            blocks = page.get_text("blocks")
            sorted_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
            for b in sorted_blocks:
                if len(b) > 4 and b[4].strip():
                    text_content.append(b[4].strip())
        doc.close()
    except Exception:
        text_content = []

    full_text = "\n".join(text_content).strip()

    # Check garbage symbol ratio and apply legacy font conversion
    g_ratio = calculate_garbage_ratio(full_text)
    if g_ratio > 0.02:
        full_text = convert_legacy_sinhala_fonts(full_text)
        extraction_method = "pymupdf_legacy_converted"

    # Secondary check: pdfplumber fallback if PyMuPDF extracted minimal text
    if len(full_text) < 50:
        try:
            with pdfplumber.open(filepath) as pdf:
                plumber_text = []
                for page in pdf.pages:
                    txt = page.extract_text(layout=True)
                    if txt:
                        plumber_text.append(txt)
                if plumber_text:
                    full_text = "\n".join(plumber_text).strip()
                    extraction_method = "pdfplumber"
                    if calculate_garbage_ratio(full_text) > 0.02:
                        full_text = convert_legacy_sinhala_fonts(full_text)
                        extraction_method = "pdfplumber_legacy_converted"
        except Exception:
            pass

    # Tertiary check: OCR fallback via pytesseract using PyMuPDF pixmaps (only if Tesseract binary is installed)
    if IS_TESSERACT_AVAILABLE and (len(full_text) < 50 or calculate_garbage_ratio(full_text) > 0.15):
        try:
            doc = pymupdf.open(filepath)
            ocr_text = []
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                try:
                    txt = pytesseract.image_to_string(img, config="-l eng+sin")
                except Exception:
                    txt = pytesseract.image_to_string(img, config="-l eng")

                if txt.strip():
                    ocr_text.append(txt.strip())
            doc.close()
            if ocr_text:
                full_text = "\n".join(ocr_text).strip()
                extraction_method = "pytesseract_ocr"
        except Exception:
            pass

    return full_text, extraction_method



def clean_structural_noise(text: str) -> str:
    """RegEx cleaning for legal headers, footers, page numbers, stamps, and formatting artifacts."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    header_footer_patterns = [
        r"(?i)IN THE COURT OF APPEAL OF THE DEMOCRATIC SOCIALIST REPUBLIC OF SRI LANKA",
        r"(?i)IN THE SUPREME COURT OF THE DEMOCRATIC SOCIALIST REPUBLIC OF SRI LANKA",
        r"(?i)PRINTED AT THE DEPARTMENT OF GOVERNMENT PRINTING, SRI LANKA",
        r"(?i)Page\s+\d+\s+of\s+\d+",
        r"(?i)Page\s+\d+",
        r"^\s*-\s*\d+\s*-\s*$",
        r"^\s*\d+\s*$",
        r"^\s*\[\s*\d+\s*\]\s*$",
        r"http[s]?://\S+",
        r"www\.\S+",
    ]

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line_str = line.strip()

        is_noise = False
        for pat in header_footer_patterns:
            if re.search(pat, line_str):
                is_noise = True
                break

        if not is_noise:
            cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def extract_verdict(text: str) -> tuple[str, str]:
    """Extract judicial outcome / verdict from case text."""
    if not text:
        return "Undetermined", ""

    footer_text = text[-2500:] if len(text) > 2500 else text
    footer_lower = footer_text.lower()

    for pattern, label in VERDICT_PATTERNS:
        match = re.search(pattern, footer_lower, re.IGNORECASE)
        if match:
            return label, match.group(0)

    text_lower = text.lower()
    for pattern, label in VERDICT_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            return label, match.group(0)

    return "Undetermined", ""


def is_family_law_document(text: str) -> tuple[bool, list[str]]:
    """Check if document matches English, Sinhala, or Tamil Family Law keywords."""
    text_lower = text.lower()
    matched = []

    # Check English keywords
    for kw in FAMILY_LAW_KEYWORDS_ENGLISH:
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            matched.append(kw)

    # Check Sinhala and Tamil keywords
    for kw in FAMILY_LAW_KEYWORDS_SINHALA + FAMILY_LAW_KEYWORDS_TAMIL:
        if kw in text:
            matched.append(kw)

    return len(matched) > 0, matched


def process_pdf_task(task_args: tuple[str, str, str]) -> dict:
    """Worker task to process a single unique PDF."""
    pdf_path_str, file_hash, rel_path = task_args
    pdf_path = Path(pdf_path_str)

    raw_text, method = extract_text_from_pdf(pdf_path)

    if not raw_text or len(raw_text) < 20:
        return {
            "status": "empty",
            "pdf_path_str": pdf_path_str,
            "filename": pdf_path.name,
            "relative_path": rel_path,
            "method": method,
            "sha256": file_hash
        }

    clean_text = clean_structural_noise(raw_text)
    doc_id = f"{pdf_path.stem}_{file_hash[:8]}"

    parent_dir_name = pdf_path.parent.name.lower()
    is_statutory = "acts" in parent_dir_name or "bills" in parent_dir_name or "acts" in rel_path.lower() or "bills" in rel_path.lower()
    is_fam_law, matched_kws = is_family_law_document(clean_text)

    verdict, matched_snippet = extract_verdict(clean_text) if (is_fam_law and not is_statutory) else ("N/A", "")

    return {
        "status": "success",
        "doc_id": doc_id,
        "filename": pdf_path.name,
        "relative_path": rel_path,
        "method": method,
        "clean_text": clean_text,
        "is_statutory": is_statutory,
        "is_family_law": is_fam_law,
        "keywords": matched_kws,
        "verdict": verdict,
        "matched_snippet": matched_snippet,
        "char_length": len(clean_text),
        "sha256": file_hash
    }


def main():
    print("=" * 80)
    print("SLLIP DATA PIPELINE - STEP 1: LEGACY FONT & TRILINGUAL PDF PROCESSOR (8 CORES)")
    print("=" * 80)

    if not RAW_DATASET_DIR.exists():
        print(f"Error: Raw dataset directory '{RAW_DATASET_DIR}' not found.")
        sys.exit(1)

    all_pdf_files = list(RAW_DATASET_DIR.rglob("*.pdf"))
    total_pdf_count = len(all_pdf_files)
    print(f"Found {total_pdf_count:,} total PDF files across '{RAW_DATASET_DIR.name}'.\n")

    print("Step 1.1: Fast SHA-256 File Deduplication...")
    hashes_seen = {}
    duplicate_count = 0
    unique_tasks = []

    for pdf_path in tqdm(all_pdf_files, desc="Deduplicating PDFs", unit="file"):
        pdf_str = str(pdf_path)
        sha256_hash = hashlib.sha256()
        try:
            with open(pdf_path, "rb") as f:
                for byte_block in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(byte_block)
            file_hash = sha256_hash.hexdigest()
        except Exception:
            continue

        rel_path = str(pdf_path.relative_to(RAW_DATASET_DIR))
        if file_hash in hashes_seen:
            duplicate_count += 1
        else:
            hashes_seen[file_hash] = rel_path
            unique_tasks.append((pdf_str, file_hash, rel_path))

    unique_count = len(unique_tasks)
    print(f"\nDeduplication Complete: {unique_count:,} Unique PDFs, {duplicate_count:,} Duplicate PDFs removed.\n")

    num_workers = min(os.cpu_count() or 4, 8)
    print(f"Step 1.2: Parallel Layout, Legacy Conversion & OCR Text Extraction across {num_workers} CPU cores...")
    extracted_count = 0
    empty_count = 0

    family_law_count = 0
    statutory_count = 0
    other_cases_count = 0

    extraction_methods = {}
    ground_truth = {}
    extraction_metadata = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results_generator = executor.map(process_pdf_task, unique_tasks, chunksize=20)
        for res in tqdm(results_generator, total=unique_count, desc="Extracting & Cleaning", unit="file"):
            method = res.get("method", "failed")
            extraction_methods[method] = extraction_methods.get(method, 0) + 1

            if res["status"] == "empty":
                empty_count += 1
                continue

            extracted_count += 1
            doc_id = res["doc_id"]
            clean_text = res["clean_text"]

            if res["is_statutory"]:
                statutory_count += 1
                out_path = STATUTORY_ACTS_TXT_DIR / f"{doc_id}.txt"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(clean_text)
            elif res["is_family_law"]:
                family_law_count += 1
                out_path = FAMILY_LAW_TXT_DIR / f"{doc_id}.txt"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(clean_text)

                ground_truth[doc_id] = {
                    "doc_id": doc_id,
                    "filename": res["filename"],
                    "relative_path": res["relative_path"],
                    "verdict": res["verdict"],
                    "matched_snippet": res["matched_snippet"],
                    "keywords_matched": res["keywords"],
                    "char_length": res["char_length"],
                    "sha256": res["sha256"]
                }
            else:
                other_cases_count += 1

            extraction_metadata.append({
                "doc_id": doc_id,
                "filename": res["filename"],
                "relative_path": res["relative_path"],
                "extraction_method": method,
                "is_family_law": res["is_family_law"],
                "is_statutory": res["is_statutory"],
                "keywords": res["keywords"],
                "char_length": res["char_length"],
                "sha256": res["sha256"]
            })

    # Save Ground Truth JSON
    with open(GROUND_TRUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    # Save Extraction Metadata JSON
    with open(EXTRACTION_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(extraction_metadata, f, indent=2, ensure_ascii=False)

    # Compute verdict distribution statistics
    verdict_stats = {}
    for entry in ground_truth.values():
        v = entry["verdict"]
        verdict_stats[v] = verdict_stats.get(v, 0) + 1

    print("\n" + "=" * 80)
    print("STEP 1: EXTRACTION, LEGACY CONVERSION & CLEANING SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total PDFs Scanned:                {total_pdf_count:,}")
    print(f"Duplicates Removed (SHA-256):     {duplicate_count:,}")
    print(f"Unique PDFs Processed:            {unique_count:,}")
    print(f"Successfully Extracted PDFs:       {extracted_count:,}")
    print(f"Empty/Unextractable PDFs:         {empty_count:,}")
    print("-" * 50)
    print("EXTRACTION METHODS BREAKDOWN:")
    for method, count in extraction_methods.items():
        print(f"  - {method:30s}: {count:,}")
    print("-" * 50)
    print("CORPUS DOMAIN ISOLATION BREAKDOWN:")
    print(f"  - Target Family Law Cases:      {family_law_count:,}")
    print(f"  - Statutory Acts & Bills:       {statutory_count:,}")
    print(f"  - Other Non-Family Law Cases:   {other_cases_count:,}")
    print("-" * 50)
    print("FAMILY LAW GROUND TRUTH VERDICT DISTRIBUTION:")
    for verdict, count in sorted(verdict_stats.items(), key=lambda x: x[1], reverse=True):
        pct = (count / family_law_count * 100) if family_law_count > 0 else 0
        print(f"  - {verdict:25s}: {count:5,} ({pct:5.1f}%)")
    print("=" * 80)
    print(f"Saved Ground Truth to:               {GROUND_TRUTH_FILE}")
    print(f"Saved Family Law Clean Text files:   {FAMILY_LAW_TXT_DIR}")
    print(f"Saved Statutory Acts Clean Text:     {STATUTORY_ACTS_TXT_DIR}")
    print("Step 1 Execution Completed Successfully!")


if __name__ == "__main__":
    main()
