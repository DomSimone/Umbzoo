"""
SQL Database for Q&A vector storage and retrieval.
Stores documents, Q&A pairs with vector embeddings for semantic search.
"""
import os
import json
import sqlite3
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from sentence_transformers import SentenceTransformer

from config import DB_PATH, TOPICS, AFRICAN_COUNTRIES


class QADatabase:
    """
    SQLite database with vector embeddings for Q&A storage and retrieval.
    Stores documents, topics, countries, questions, and answers with
    vector embeddings for semantic similarity search.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

        # Load embedding model
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            print("Loading sentence transformer for embeddings...")
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def _create_tables(self):
        """Create database schema."""
        cursor = self.conn.cursor()

        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                file_type TEXT,
                title TEXT,
                content TEXT NOT NULL,
                content_hash TEXT UNIQUE,
                char_count INTEGER DEFAULT 0,
                word_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Topics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                parent_topic TEXT
            )
        """)

        # Countries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS countries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                region TEXT,
                code TEXT
            )
        """)

        # Document-Topics mapping
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_topics (
                document_id INTEGER,
                topic_id INTEGER,
                relevance_score REAL DEFAULT 1.0,
                PRIMARY KEY (document_id, topic_id),
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
            )
        """)

        # Document-Countries mapping
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_countries (
                document_id INTEGER,
                country_id INTEGER,
                PRIMARY KEY (document_id, country_id),
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                FOREIGN KEY (country_id) REFERENCES countries(id) ON DELETE CASCADE
            )
        """)

        # Q&A pairs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qa_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                question_type TEXT DEFAULT 'factual',
                difficulty TEXT DEFAULT 'medium',
                reasoning_type TEXT,
                document_id INTEGER,
                topic_id INTEGER,
                country_id INTEGER,
                vector_range_start REAL,
                vector_range_end REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id),
                FOREIGN KEY (topic_id) REFERENCES topics(id),
                FOREIGN KEY (country_id) REFERENCES countries(id)
            )
        """)

        # Vector embeddings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vector_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                dimension INTEGER DEFAULT 384,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_qa_topic ON qa_pairs(topic_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_qa_country ON qa_pairs(country_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_qa_type ON qa_pairs(question_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vec_entity ON vector_embeddings(entity_type, entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(content_hash)")

        self.conn.commit()

    def insert_topics(self):
        """Insert predefined topics into the database."""
        cursor = self.conn.cursor()
        for topic_key, topic_info in TOPICS.items():
            cursor.execute(
                "INSERT OR IGNORE INTO topics (name, description) VALUES (?, ?)",
                (topic_key, topic_info["description"]),
            )
        self.conn.commit()

    def insert_countries(self):
        """Insert African countries into the database."""
        cursor = self.conn.cursor()
        for country in AFRICAN_COUNTRIES:
            cursor.execute(
                "INSERT OR IGNORE INTO countries (name) VALUES (?)",
                (country,),
            )
        self.conn.commit()

    def insert_document(self, source: str, content: str, file_type: str = None,
                        title: str = None, topics: List[str] = None,
                        countries: List[str] = None) -> int:
        """Insert a document and return its ID."""
        import hashlib
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()

        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO documents 
                (source, file_type, title, content, content_hash, char_count, word_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (source, file_type, title, content, content_hash,
                  len(content), len(content.split())))
            doc_id = cursor.lastrowid

            # If document already exists, get its ID
            if doc_id == 0:
                cursor.execute("SELECT id FROM documents WHERE content_hash = ?", (content_hash,))
                row = cursor.fetchone()
                return row[0] if row else -1

            # Link topics
            if topics:
                for topic in topics:
                    cursor.execute("SELECT id FROM topics WHERE name = ?", (topic,))
                    row = cursor.fetchone()
                    if row:
                        cursor.execute(
                            "INSERT OR IGNORE INTO document_topics (document_id, topic_id) VALUES (?, ?)",
                            (doc_id, row[0]),
                        )

            # Link countries
            if countries:
                for country in countries:
                    cursor.execute("SELECT id FROM countries WHERE name = ?", (country,))
                    row = cursor.fetchone()
                    if row:
                        cursor.execute(
                            "INSERT OR IGNORE INTO document_countries (document_id, country_id) VALUES (?, ?)",
                            (doc_id, row[0]),
                        )

            self.conn.commit()
            return doc_id
        except Exception as e:
            self.conn.rollback()
            print(f"Error inserting document: {e}")
            return -1

    def insert_qa(self, question: str, answer: str, question_type: str = "factual",
                  difficulty: str = "medium", reasoning_type: str = None,
                  document_id: int = None, topic_id: int = None,
                  country_id: int = None) -> int:
        """Insert a Q&A pair and compute its vector embedding."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO qa_pairs 
                (question, answer, question_type, difficulty, reasoning_type, 
                 document_id, topic_id, country_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (question, answer, question_type, difficulty, reasoning_type,
                  document_id, topic_id, country_id))
            qa_id = cursor.lastrowid

            # Compute and store vector embedding
            combined = f"{question} {answer}"
            self._store_embedding("qa", qa_id, combined)

            self.conn.commit()
            return qa_id
        except Exception as e:
            self.conn.rollback()
            print(f"Error inserting Q&A: {e}")
            return -1

    def _store_embedding(self, entity_type: str, entity_id: int, text: str):
        """Compute and store vector embedding."""
        embedding = self.embedder.encode(text, normalize_embeddings=True)
        embedding_bytes = embedding.astype(np.float32).tobytes()
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO vector_embeddings 
            (entity_type, entity_id, embedding, dimension)
            VALUES (?, ?, ?, ?)
        """, (entity_type, entity_id, embedding_bytes, len(embedding)))
        self.conn.commit()

    def search_similar(self, query: str, entity_type: str = "qa",
                       top_k: int = 5, topic_id: int = None,
                       country_id: int = None) -> List[Dict]:
        """Search for similar Q&A pairs by vector similarity."""
        query_embedding = self.embedder.encode(query, normalize_embeddings=True)

        cursor = self.conn.cursor()

        # Build query with optional filters
        sql = """
            SELECT v.id, v.entity_id, v.embedding, q.question, q.answer, q.difficulty,
                   q.question_type, q.reasoning_type
            FROM vector_embeddings v
            JOIN qa_pairs q ON v.entity_id = q.id
            WHERE v.entity_type = ?
        """
        params = [entity_type]

        if topic_id:
            sql += " AND q.topic_id = ?"
            params.append(topic_id)
        if country_id:
            sql += " AND q.country_id = ?"
            params.append(country_id)

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        # Compute cosine similarity
        results = []
        for row in rows:
            db_embedding = np.frombuffer(row[2], dtype=np.float32)
            similarity = float(np.dot(query_embedding, db_embedding))
            results.append({
                "id": row[1],
                "question": row[3],
                "answer": row[4],
                "difficulty": row[5],
                "question_type": row[6],
                "reasoning_type": row[7],
                "similarity": similarity,
            })

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def get_qa_by_topic(self, topic_name: str, limit: int = 100) -> List[Dict]:
        """Retrieve Q&A pairs by topic."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT q.id, q.question, q.answer, q.difficulty, q.question_type,
                   q.reasoning_type, t.name as topic, c.name as country
            FROM qa_pairs q
            LEFT JOIN topics t ON q.topic_id = t.id
            LEFT JOIN countries c ON q.country_id = c.id
            WHERE t.name = ?
            ORDER BY q.created_at DESC
            LIMIT ?
        """, (topic_name, limit))
        rows = cursor.fetchall()
        return [
            {
                "id": r[0], "question": r[1], "answer": r[2],
                "difficulty": r[3], "question_type": r[4],
                "reasoning_type": r[5], "topic": r[6], "country": r[7],
            }
            for r in rows
        ]

    def get_qa_by_country(self, country_name: str, limit: int = 100) -> List[Dict]:
        """Retrieve Q&A pairs by country."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT q.id, q.question, q.answer, q.difficulty, q.question_type,
                   q.reasoning_type, t.name as topic, c.name as country
            FROM qa_pairs q
            LEFT JOIN topics t ON q.topic_id = t.id
            LEFT JOIN countries c ON q.country_id = c.id
            WHERE c.name = ?
            ORDER BY q.created_at DESC
            LIMIT ?
        """, (country_name, limit))
        rows = cursor.fetchall()
        return [
            {
                "id": r[0], "question": r[1], "answer": r[2],
                "difficulty": r[3], "question_type": r[4],
                "reasoning_type": r[5], "topic": r[6], "country": r[7],
            }
            for r in rows
        ]

    def get_stats(self) -> Dict:
        """Get database statistics."""
        cursor = self.conn.cursor()
        stats = {}
        cursor.execute("SELECT COUNT(*) FROM documents")
        stats["documents"] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM qa_pairs")
        stats["qa_pairs"] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM vector_embeddings")
        stats["embeddings"] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM topics")
        stats["topics"] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM countries")
        stats["countries"] = cursor.fetchone()[0]
        cursor.execute("SELECT SUM(char_count) FROM documents")
        stats["total_chars"] = cursor.fetchone()[0] or 0
        return stats

    def close(self):
        """Close the database connection."""
        self.conn.close()


if __name__ == "__main__":
    db = QADatabase()
    db.insert_topics()
    db.insert_countries()
    print("Database initialized with topics and countries.")
    print("Stats:", db.get_stats())
    db.close()