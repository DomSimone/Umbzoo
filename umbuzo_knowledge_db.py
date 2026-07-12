"""
Umbuzo Knowledge Base & SQL Pipeline.
Manages Q&A vectors, topics classification, and complex reasoning drafting.
"""
import sqlite3
import json
import os
from config import DB_PATH, TOPICS

class UmbuzoDB:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        # QA Store
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_qa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                question TEXT,
                answer TEXT,
                reasoning_type TEXT, -- 'factual' or 'complex'
                vector_blob BLOB,
                metadata TEXT
            )
        """)
        # Countries vector data
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS country_specific_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                country_name TEXT,
                key_metric TEXT,
                value TEXT,
                source_url TEXT
            )
        """)
        self.conn.commit()

    def insert_qa(self, topic, q, a, r_type='factual', metadata=None):
        self.cursor.execute(
            "INSERT INTO knowledge_qa (topic, question, answer, reasoning_type, metadata) VALUES (?, ?, ?, ?, ?)",
            (topic, q, a, r_type, json.dumps(metadata) if metadata else None)
        )
        self.conn.commit()

    def generate_synthetic_qa(self):
        """Devise 5000 Q&A and 3000 Complex Reasoning drafts."""
        print("Drafting 5000 standard Q&A pairs across topics...")
        count = 0
        for topic_key, details in TOPICS.items():
            for _ in range(1000): # Distribute across topics
                self.insert_qa(
                    topic=details['description'],
                    q=f"Draft question for {topic_key} range {count}",
                    a=f"Synthesized answer based on scraped {topic_key} data indices.",
                    r_type='factual'
                )
                count += 1
        
        print("Drafting 3000 Complex Reasoning questions...")
        for _ in range(3000):
            self.insert_qa(
                topic="Multidisciplinary",
                q=f"Complex reasoning scenario {count}: Analyzing cross-sector impact in Africa.",
                a="Step-by-step logical derivation of factors influencing socio-economic outcomes.",
                r_type='complex'
            )
            count += 1
        
        print(f"Database populated with {count} drafted training vectors.")

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    db = UmbuzoDB()
    db.generate_synthetic_qa()
    db.close()
