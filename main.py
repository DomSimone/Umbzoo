"""
Africa-focused LLM Data Processing Pipeline.
Orchestrates data extraction, tokenization, Q&A generation, and training.
"""
import os
import sys
import json
import argparse
from typing import Dict, List, Optional
from datetime import datetime

from config import (
    DATA_DIR, MODELS_DIR, DB_PATH, EXTRACTED_TEXT_DIR,
    TOPICS, AFRICAN_COUNTRIES, model_cfg, data_cfg,
)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def step_extract_pdfs():
    """Step 1: Extract text from PDF documents."""
    print_header("STEP 1: PDF/TEXT EXTRACTION")
    from data_extractor import DataExtractor
    extractor = DataExtractor()
    results = extractor.process_directory()
    total_chars = len(extractor.get_all_extracted_text())
    print(f"  Files processed: {len(results)}")
    print(f"  Total corpus size: {total_chars:,} characters")
    print(f"  Extracted text saved to: {EXTRACTED_TEXT_DIR}")
    return results


def step_scrape_web():
    """Step 2: Scrape web sources."""
    print_header("STEP 2: WEB SCRAPING")
    from scraper import WebScraper
    scraper = WebScraper()
    content = scraper.scrape_all_sources(max_pages=5)
    scraper.save_scraped_content(content)
    print(f"  Items scraped: {len(content)}")
    return content


def step_train_tokenizer():
    """Step 3: Train GPT-2 style BPE tokenizer."""
    print_header("STEP 3: GPT-2 TOKENIZER TRAINING")
    from tokenizer_pipeline import GPT2TokenizerPipeline
    pipeline = GPT2TokenizerPipeline()
    tokenizer = pipeline.train_tokenizer()
    pipeline.show_sample_tokenization()
    print(f"  Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"  Tokenizer saved to: {pipeline.tokenizer_dir}")
    return pipeline


def step_generate_qa():
    """Step 4: Generate Q&A dataset with complex reasoning."""
    print_header("STEP 4: Q&A DATASET GENERATION")
    from qa_generator import QAGenerator
    generator = QAGenerator()
    result = generator.generate_comprehensive_dataset()
    print(f"  Total Q&A pairs generated: {result['total_qa']}")
    print(f"  Topics covered: {result['topics_covered']}")
    print(f"  Output saved to: {result['output_dir']}")
    return result


def step_init_database():
    """Step 5: Initialize SQLite database and insert data."""
    print_header("STEP 5: SQL DATABASE INITIALIZATION")
    from sql_database import QADatabase
    db = QADatabase()
    db.insert_topics()
    db.insert_countries()
    stats = db.get_stats()
    print(f"  Database path: {DB_PATH}")
    print(f"  Topics: {stats['topics']}")
    print(f"  Countries: {stats['countries']}")
    print(f"  Documents: {stats['documents']}")
    print(f"  Q&A Pairs: {stats['qa_pairs']}")

    # Insert generated Q&A pairs into database
    qa_dir = os.path.join(DATA_DIR, "qa_datasets")
    if os.path.isdir(qa_dir):
        import json
        for fname in os.listdir(qa_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(qa_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    qa_list = json.load(f)
                for qa in qa_list[:50]:  # Insert first 50 per file
                    topic_name = qa.get("topic")
                    country_name = qa.get("country")
                    topic_id = None
                    country_id = None

                    if topic_name:
                        cursor = db.conn.cursor()
                        cursor.execute("SELECT id FROM topics WHERE name = ?", (topic_name,))
                        row = cursor.fetchone()
                        if row:
                            topic_id = row[0]

                    if country_name:
                        cursor = db.conn.cursor()
                        cursor.execute("SELECT id FROM countries WHERE name = ?", (country_name,))
                        row = cursor.fetchone()
                        if row:
                            country_id = row[0]

                    db.insert_qa(
                        question=qa["question"],
                        answer=qa["answer"],
                        question_type=qa.get("question_type", "factual"),
                        difficulty=qa.get("difficulty", "medium"),
                        reasoning_type=qa.get("reasoning_type"),
                        topic_id=topic_id,
                        country_id=country_id,
                    )

        stats = db.get_stats()
        print(f"  After insertion - Q&A Pairs: {stats['qa_pairs']}")

    db.close()
    return stats


def step_create_dataloaders():
    """Step 6: Create DataLoaders with 80/20 split."""
    print_header("STEP 6: DATA LOADER CREATION")
    from tokenizer_pipeline import GPT2TokenizerPipeline
    from data_loader import DataLoaderFactory, inspect_batch

    # Load tokenizer
    pipeline = GPT2TokenizerPipeline()
    try:
        tokenizer = pipeline.load_tokenizer()
    except FileNotFoundError:
        print("Tokenizer not found. Training first...")
        pipeline.train_tokenizer()
        tokenizer = pipeline.load_tokenizer()

    # Create data loaders
    factory = DataLoaderFactory(tokenizer=tokenizer)

    # Try Q&A data first
    qa_data = factory.load_qa_data()
    if qa_data:
        print(f"Loaded {len(qa_data)} Q&A pairs")
        train_loader, val_loader = factory.create_qa_dataloaders(qa_data)
    else:
        print("No Q&A data found. Using synthetic text data...")
        train_loader, val_loader = factory.create_text_dataloaders()

    # Inspect a sample batch
    batch = factory.get_sample_batch(train_loader)
    if batch:
        inspect_batch(batch)

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    return train_loader, val_loader


def test_nlu_classification():
    """Test Natural Language Understanding classification."""
    print_header("NLU CLASSIFICATION TEST")

    test_queries = [
        "What is the GDP growth rate of South Africa?",
        "How has urbanization affected Nairobi?",
        "Compare Nigeria and Kenya voting patterns",
        "What are the healthcare challenges in Ethiopia?",
        "Explain the relationship between education and income in Ghana",
        "What is Agenda 2063?",
        "How does climate change affect agriculture in Africa?",
        "What are the main exports of Angola?",
        "Analyze the impact of mobile banking on financial inclusion",
        "Compare economic policies across East African countries",
    ]

    from sql_database import QADatabase
    from qa_generator import QAGenerator

    db = QADatabase()
    generator = QAGenerator()

    print("\nClassifying test queries by topic and detecting countries...\n")
    for query in test_queries:
        topic = generator._classify_text(query)
        country = generator._detect_country(query)
        topic_desc = TOPICS.get(topic, {}).get("description", "general") if topic else "general"
        print(f"  Query: {query}")
        print(f"    -> Topic: {topic_desc} ({topic})")
        print(f"    -> Country: {country or 'General Africa'}")

        # Search for similar Q&A in database
        try:
            similar = db.search_similar(query, top_k=2)
            if similar:
                print(f"    -> Similar QA (top): {similar[0]['question'][:80]}...")
        except Exception:
            pass
        print()

    db.close()


def run_pipeline(steps: List[str] = None):
    """Run the full pipeline or selected steps."""
    if steps is None:
        steps = ["extract", "scrape", "tokenizer", "qa", "database", "dataloaders", "test"]

    results = {}

    if "extract" in steps:
        results["extraction"] = step_extract_pdfs()

    if "scrape" in steps:
        results["scraping"] = step_scrape_web()

    if "tokenizer" in steps:
        results["tokenizer"] = step_train_tokenizer()

    if "qa" in steps:
        results["qa"] = step_generate_qa()

    if "database" in steps:
        results["database"] = step_init_database()

    if "dataloaders" in steps:
        results["dataloaders"] = step_create_dataloaders()

    if "test" in steps:
        test_nlu_classification()

    if "train" in steps:
        print_header("STEP: TRAINING")
        from train import main as train_main
        sys.argv = [sys.argv[0], "--epochs", "1", "--batch_size", "2", "--train_only"]
        train_main()

    return results


def print_summary(results: Dict):
    """Print a summary of all pipeline results."""
    print_header("PIPELINE SUMMARY")
    for step, result in results.items():
        if isinstance(result, list):
            print(f"  {step}: {len(result)} items processed")
        elif isinstance(result, dict):
            print(f"  {step}: {json.dumps(result, indent=4)}")
        elif hasattr(result, '__class__'):
            print(f"  {step}: Completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Africa LLM Data Pipeline")
    parser.add_argument("--steps", type=str, nargs="+",
                        choices=["extract", "scrape", "tokenizer", "qa",
                                 "database", "dataloaders", "test", "train", "all"],
                        default=["test"],
                        help="Pipeline steps to run")
    parser.add_argument("--quick-test", action="store_true",
                        help="Run quick verification test")
    args = parser.parse_args()

    if args.quick_test or "test" in args.steps:
        test_nlu_classification()
    elif "all" in args.steps:
        results = run_pipeline([
            "extract", "scrape", "tokenizer", "qa", "database",
            "dataloaders", "test"
        ])
        print_summary(results)
    else:
        results = run_pipeline(args.steps)
        print_summary(results)

    print("\n" + "=" * 70)
    print("  Pipeline execution complete!")
    print("=" * 70)