"""
Data Extractor: Extracts text from PDFs, TXT, CSV files and web sources.
Uses PyMuPDF for PDF parsing and handles various document formats.
"""
import os
import json
import csv
import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm

from config import TEXTS_DIR, EXTRACTED_TEXT_DIR, TOPICS, AFRICAN_COUNTRIES, COUNTRY_KEYWORDS


class DataExtractor:
    """Extracts and processes text from various document formats."""

    def __init__(self):
        self.extracted_dir = EXTRACTED_TEXT_DIR
        os.makedirs(self.extracted_dir, exist_ok=True)

    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from a PDF file using PyMuPDF."""
        text_parts = []
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(page_text)
            doc.close()
        except Exception as e:
            print(f"Error extracting {pdf_path}: {e}")
            return ""
        return "\n\n".join(text_parts)

    def extract_txt_text(self, txt_path: str) -> str:
        """Extract text from a TXT file."""
        try:
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {txt_path}: {e}")
            return ""

    def extract_csv_text(self, csv_path: str) -> str:
        """Extract text from a CSV file, converting rows to readable text."""
        text_parts = []
        try:
            with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                if headers:
                    text_parts.append(" | ".join(headers))
                for row in reader:
                    text_parts.append(" | ".join(row))
        except Exception as e:
            print(f"Error reading CSV {csv_path}: {e}")
            return ""
        return "\n".join(text_parts)

    def extract_file(self, file_path: str) -> Tuple[str, str]:
        """
        Extract text from a file based on its extension.
        Returns (text, file_type).
        """
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return self.extract_pdf_text(file_path), "pdf"
        elif ext == ".txt":
            return self.extract_txt_text(file_path), "txt"
        elif ext == ".csv":
            return self.extract_csv_text(file_path), "csv"
        else:
            return "", "unknown"

    def classify_topic(self, text: str) -> List[str]:
        """Classify text into one or more topics based on keyword matching."""
        text_lower = text.lower()
        matched_topics = []
        for topic_key, topic_info in TOPICS.items():
            keywords = topic_info["keywords"]
            if any(kw.lower() in text_lower for kw in keywords):
                matched_topics.append(topic_key)
        return matched_topics if matched_topics else ["general"]

    def detect_countries(self, text: str) -> List[str]:
        """Detect which African countries are mentioned in the text."""
        text_lower = text.lower()
        detected = []
        for country in AFRICAN_COUNTRIES:
            if country.lower() in text_lower:
                detected.append(country)
            elif country in COUNTRY_KEYWORDS:
                if any(kw.lower() in text_lower for kw in COUNTRY_KEYWORDS[country]):
                    detected.append(country)
        return detected

    def extract_metadata(self, text: str, file_path: str, file_type: str) -> Dict:
        """Extract metadata from text content."""
        return {
            "source_file": file_path,
            "file_type": file_type,
            "char_count": len(text),
            "word_count": len(text.split()),
            "topics": self.classify_topic(text),
            "countries": self.detect_countries(text),
            "file_hash": hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest(),
        }

    def process_directory(self, directory: str = None) -> List[Dict]:
        """Process all supported files in a directory."""
        if directory is None:
            directory = TEXTS_DIR

        if not os.path.isdir(directory):
            print(f"Directory not found: {directory}")
            return []

        supported_exts = {".pdf", ".txt", ".csv"}
        files = []
        for f in os.listdir(directory):
            if Path(f).suffix.lower() in supported_exts:
                files.append(os.path.join(directory, f))

        if not files:
            print(f"No supported files found in {directory}")
            return []

        results = []
        for file_path in tqdm(files, desc="Extracting files"):
            text, file_type = self.extract_file(file_path)
            if not text.strip():
                continue

            # Save extracted text
            base_name = Path(file_path).stem
            out_path = os.path.join(self.extracted_dir, f"{base_name}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)

            metadata = self.extract_metadata(text, file_path, file_type)
            metadata["extracted_text_path"] = out_path
            results.append(metadata)

            print(f"  Extracted: {base_name} ({len(text)} chars, topics: {metadata['topics']}, countries: {metadata['countries']})")

        # Save metadata index
        meta_path = os.path.join(self.extracted_dir, "metadata_index.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\nProcessed {len(results)} files. Metadata saved to {meta_path}")
        return results

    def get_all_extracted_text(self) -> str:
        """Concatenate all extracted text into a single corpus."""
        all_text = []
        for fname in os.listdir(self.extracted_dir):
            if fname.endswith(".txt") and fname != "metadata_index.json":
                fpath = os.path.join(self.extracted_dir, fname)
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    all_text.append(f.read())
        return "\n\n".join(all_text)


if __name__ == "__main__":
    extractor = DataExtractor()
    results = extractor.process_directory()
    print(f"\nTotal files processed: {len(results)}")
    print(f"Total corpus size: {len(extractor.get_all_extracted_text())} chars")