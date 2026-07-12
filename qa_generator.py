"""
Q&A Generator: Creates factual and complex reasoning Q&A pairs
from extracted documents for LLM training and refinement.
"""
import os
import json
import random
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from config import TOPICS, AFRICAN_COUNTRIES, REASONING_TEMPLATES, EXTRACTED_TEXT_DIR


class QAGenerator:
    """
    Generates Q&A pairs from document content for LLM training.
    Produces both factual and complex reasoning questions.
    """

    def __init__(self):
        self.topics = TOPICS
        self.countries = AFRICAN_COUNTRIES
        self.templates = REASONING_TEMPLATES
        random.seed(42)

    def extract_sentences(self, text: str, max_sentences: int = 1000) -> List[str]:
        """Extract sentences from text."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20][:max_sentences]

    def generate_factual_qa(self, text: str, source: str = None) -> List[Dict]:
        """
        Generate factual Q&A pairs from document text.
        Uses sentence patterns to create question-answer pairs.
        """
        sentences = self.extract_sentences(text)
        qa_pairs = []

        # Pattern-based Q&A generation
        patterns = [
            # (keyword, question_template, answer_extractor)
            (" is ", "What is {subject}?", None),
            (" are ", "What are {subject}?", None),
            (" was established in ", "When was {subject} established?", None),
            (" was founded in ", "When was {subject} founded?", None),
            (" was created in ", "When was {subject} created?", None),
            (" was adopted in ", "When was {subject} adopted?", None),
            (" was launched in ", "When was {subject} launched?", None),
            (" aims to ", "What is the aim of {subject}?", None),
            (" seeks to ", "What does {subject} seek to do?", None),
            (" promotes ", "What does {subject} promote?", None),
            (" focuses on ", "What does {subject} focus on?", None),
            (" is located in ", "Where is {subject} located?", None),
            (" is headquartered in ", "Where is {subject} headquartered?", None),
            (" consists of ", "What does {subject} consist of?", None),
            (" includes ", "What does {subject} include?", None),
            (" provides ", "What does {subject} provide?", None),
            (" supports ", "What does {subject} support?", None),
            (" addresses ", "What does {subject} address?", None),
            (" strengthens ", "What does {subject} strengthen?", None),
            (" improves ", "What does {subject} improve?", None),
            (" transforms ", "How does {subject} transform?", None),
            (" is essential for ", "Why is {subject} essential?", None),
            (" is critical for ", "Why is {subject} critical?", None),
            (" is important for ", "Why is {subject} important?", None),
            (" plays a key role in ", "What role does {subject} play?", None),
            (" contributes to ", "How does {subject} contribute?", None),
            (" is a major ", "What is {subject}?", None),
            (" is an important ", "What is {subject}?", None),
            (" is the leading ", "What is {subject}?", None),
        ]

        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword, q_template, _ in patterns:
                if keyword in sentence_lower:
                    # Extract subject (words before the keyword)
                    idx = sentence_lower.find(keyword)
                    subject = sentence[:idx].strip()
                    if subject and len(subject) < 100:
                        # Clean up subject
                        subject = subject.lstrip(" ,;:-")
                        if subject and not subject.startswith(("The ", "A ", "An ")):
                            subject = subject

                        question = q_template.format(subject=subject)
                        answer = sentence

                        # Classify topic
                        topic = self._classify_text(sentence)
                        country = self._detect_country(sentence)

                        qa_pairs.append({
                            "question": question,
                            "answer": answer,
                            "question_type": "factual",
                            "difficulty": "easy",
                            "reasoning_type": None,
                            "topic": topic,
                            "country": country,
                            "source": source,
                        })
                    break  # Only use first matching pattern per sentence

        return qa_pairs

    def generate_complex_reasoning_qa(self, text: str, source: str = None,
                                       num_questions: int = 20) -> List[Dict]:
        """
        Generate complex reasoning Q&A pairs using templates.
        These require multi-step reasoning, synthesis, or analysis.
        """
        sentences = self.extract_sentences(text)
        qa_pairs = []
        template_keys = list(self.templates.keys())

        for _ in range(min(num_questions, len(sentences) // 5 + 1)):
            template_key = random.choice(template_keys)
            template = self.templates[template_key]

            # Select random context from text
            context_sentences = random.sample(sentences, min(3, len(sentences)))
            context = " ".join(context_sentences)

            # Fill template with context-derived values
            topic = self._classify_text(context)
            country = self._detect_country(context) or random.choice(self.countries)

            # Extract key terms for template filling
            words = context.split()
            key_terms = [w for w in words if len(w) > 5 and w[0].isupper()]
            topic_a = key_terms[0] if len(key_terms) > 0 else "economic development"
            topic_b = key_terms[1] if len(key_terms) > 1 else "social progress"
            factor_a = key_terms[0] if key_terms else "governance"
            factor_b = key_terms[1] if len(key_terms) > 1 else "investment"

            # Fill template
            try:
                question = template.format(
                    topic_a=topic_a, topic_b=topic_b,
                    region=country or "Africa",
                    phenomenon=topic_a,
                    factor_a=factor_a, factor_b=factor_b,
                    topic=topic or "development",
                    policy=topic_a,
                    issue=topic_b,
                    factor_c=key_terms[2] if len(key_terms) > 2 else "technology",
                    outcome="development outcomes",
                    system_elements=f"{topic_a}, {topic_b}, and {factor_a}",
                    source_a="historical records",
                    source_b="current data",
                    historical_event="colonial period",
                    sector="economic",
                    year_start="2000",
                    year_end="2025",
                    value_a="economic growth",
                    value_b="environmental protection",
                )
            except KeyError:
                continue

            # Generate answer from context
            answer = self._generate_reasoning_answer(question, context, template_key)

            qa_pairs.append({
                "question": question,
                "answer": answer,
                "question_type": "complex_reasoning",
                "difficulty": random.choice(["medium", "hard", "expert"]),
                "reasoning_type": template_key,
                "topic": topic,
                "country": country,
                "source": source,
            })

        return qa_pairs

    def _classify_text(self, text: str) -> Optional[str]:
        """Classify text into a topic."""
        text_lower = text.lower()
        for topic_key, topic_info in self.topics.items():
            if any(kw.lower() in text_lower for kw in topic_info["keywords"]):
                return topic_key
        return None

    def _detect_country(self, text: str) -> Optional[str]:
        """Detect country mention in text."""
        text_lower = text.lower()
        for country in self.countries:
            if country.lower() in text_lower:
                return country
        return None

    def _generate_reasoning_answer(self, question: str, context: str,
                                    reasoning_type: str) -> str:
        """Generate a reasoning-based answer from context."""
        # Extract key information from context
        sentences = self.extract_sentences(context, max_sentences=5)
        key_points = [s for s in sentences if len(s) > 30][:3]

        if not key_points:
            key_points = ["Based on available data, this requires further analysis."]

        reasoning_intros = {
            "comparative_analysis": "A comparative analysis reveals several key differences and similarities. ",
            "causal_analysis": "The root causes are multi-faceted and interconnected. ",
            "predictive_modeling": "Based on historical trends and current data, the projected trajectory suggests several possible scenarios. ",
            "policy_evaluation": "Evaluation of this policy requires examining multiple dimensions of impact. ",
            "multi_factor_analysis": "Multiple factors interact in complex ways to shape this outcome. ",
            "systems_thinking": "This issue is part of a complex, interconnected system where changes in one area cascade through others. ",
            "evidence_synthesis": "Synthesizing evidence from multiple sources reveals both convergence and divergence in findings. ",
            "counterfactual_reasoning": "Counterfactual analysis suggests that alternative historical trajectories would have led to significantly different outcomes. ",
            "longitudinal_trend": "Tracing this trend over time reveals important inflection points and evolving dynamics. ",
            "ethical_dilemma": "This ethical dilemma requires balancing competing values and considering multiple stakeholder perspectives. ",
        }

        answer = reasoning_intros.get(reasoning_type, "Analysis of the available information indicates that ")
        answer += " ".join(key_points)
        answer += " This analysis is based on the available document corpus and should be supplemented with additional research for comprehensive understanding."

        return answer

    def generate_all_qa(self, text: str, source: str = None,
                        num_reasoning: int = 10) -> List[Dict]:
        """Generate both factual and complex reasoning Q&A pairs."""
        factual = self.generate_factual_qa(text, source)
        reasoning = self.generate_complex_reasoning_qa(text, source, num_reasoning)
        return factual + reasoning

    def save_qa_to_json(self, qa_pairs: List[Dict], output_path: str):
        """Save Q&A pairs to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=2)
        print(f"Saved {len(qa_pairs)} Q&A pairs to {output_path}")

    def generate_topic_specific_qa(self, topic_key: str, num_per_topic: int = 20) -> List[Dict]:
        """Generate Q&A pairs specific to a topic."""
        topic_info = self.topics.get(topic_key, {})
        subtopics = topic_info.get("subtopics", [])
        keywords = topic_info.get("keywords", [])
        description = topic_info.get("description", topic_key)

        qa_pairs = []

        # Factual questions about the topic
        for subtopic in subtopics[:3]:
            qa_pairs.append({
                "question": f"What are the key trends in {subtopic.replace('_', ' ')} in Africa?",
                "answer": f"Analysis of {subtopic.replace('_', ' ')} in Africa reveals significant "
                          f"variation across regions and countries. Key factors include economic conditions, "
                          f"policy frameworks, demographic changes, and social dynamics. "
                          f"Data from multiple sources indicates evolving patterns that require "
                          f"context-specific analysis and response strategies.",
                "question_type": "factual",
                "difficulty": "medium",
                "reasoning_type": None,
                "topic": topic_key,
                "country": None,
                "source": "generated",
            })

        # Complex reasoning questions
        for i in range(min(num_per_topic, 10)):
            country = random.choice(self.countries)
            kw1 = random.choice(keywords) if keywords else "development"
            kw2 = random.choice(keywords) if len(keywords) > 1 else "growth"

            qa_pairs.append({
                "question": f"How does {kw1} interact with {kw2} to shape "
                           f"{description.lower()} outcomes in {country}?",
                "answer": f"The interaction between {kw1} and {kw2} in {country} is complex "
                         f"and multi-dimensional. Evidence suggests that these factors "
                         f"mutually influence each other through various mechanisms including "
                         f"policy feedback loops, institutional dynamics, and socio-economic "
                         f"conditions. A comprehensive analysis requires examining both "
                         f"quantitative indicators and qualitative contextual factors specific "
                         f"to {country}'s unique historical and political landscape.",
                "question_type": "complex_reasoning",
                "difficulty": random.choice(["medium", "hard"]),
                "reasoning_type": "multi_factor_analysis",
                "topic": topic_key,
                "country": country,
                "source": "generated",
            })

        return qa_pairs

    def generate_comprehensive_dataset(self, output_dir: str = None) -> Dict:
        """Generate a comprehensive Q&A dataset covering all topics and countries."""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "data", "qa_datasets")
        os.makedirs(output_dir, exist_ok=True)

        all_qa = []

        # Generate topic-specific Q&A
        for topic_key in self.topics:
            topic_qa = self.generate_topic_specific_qa(topic_key, num_per_topic=15)
            all_qa.extend(topic_qa)

        # Generate country-specific Q&A
        for country in random.sample(self.countries, min(10, len(self.countries))):
            for topic_key in random.sample(list(self.topics.keys()), min(3, len(self.topics))):
                topic_info = self.topics[topic_key]
                kw = random.choice(topic_info["keywords"]) if topic_info["keywords"] else "development"
                all_qa.append({
                    "question": f"What are the key {topic_key.replace('_', ' ')} "
                               f"challenges and opportunities in {country}?",
                    "answer": f"{country} faces both significant challenges and opportunities "
                             f"in the area of {topic_key.replace('_', ' ')}. Key challenges include "
                             f"infrastructure gaps, resource constraints, and institutional capacity "
                             f"limitations. However, opportunities exist through policy reforms, "
                             f"international partnerships, technological innovation, and demographic "
                             f"dividends. A strategic approach that leverages local strengths while "
                             f"addressing systemic weaknesses is essential for sustainable progress.",
                    "question_type": "complex_reasoning",
                    "difficulty": "medium",
                    "reasoning_type": "policy_evaluation",
                    "topic": topic_key,
                    "country": country,
                    "source": "generated",
                })

        # Add complex reasoning questions using templates
        for template_key, template in self.templates.items():
            for _ in range(5):
                country = random.choice(self.countries)
                topic_keys = random.sample(list(self.topics.keys()), 2)
                t1 = self.topics[topic_keys[0]]
                t2 = self.topics[topic_keys[1]]
                kw1 = random.choice(t1["keywords"]) if t1["keywords"] else "development"
                kw2 = random.choice(t2["keywords"]) if t2["keywords"] else "growth"

                try:
                    question = template.format(
                        topic_a=kw1, topic_b=kw2,
                        region=country,
                        phenomenon=kw1,
                        factor_a=kw1, factor_b=kw2,
                        topic=t1["description"],
                        policy=kw1,
                        issue=kw2,
                        factor_c="technology",
                        outcome="development",
                        system_elements=f"{kw1}, {kw2}, and governance",
                        source_a="academic research",
                        source_b="policy documents",
                        historical_event="independence",
                        sector="economic",
                        year_start="2000",
                        year_end="2025",
                        value_a="economic efficiency",
                        value_b="social equity",
                    )
                    all_qa.append({
                        "question": question,
                        "answer": f"This {template_key.replace('_', ' ')} question requires "
                                 f"examining multiple dimensions of the relationship between "
                                 f"{kw1} and {kw2} in {country}. Evidence from various sources "
                                 f"suggests that the dynamics are shaped by historical context, "
                                 f"institutional frameworks, and contemporary socio-economic "
                                 f"conditions. A thorough analysis would need to consider both "
                                 f"quantitative data and qualitative insights to provide a "
                                 f"comprehensive understanding of this complex relationship.",
                        "question_type": "complex_reasoning",
                        "difficulty": "hard",
                        "reasoning_type": template_key,
                        "topic": topic_keys[0],
                        "country": country,
                        "source": "generated",
                    })
                except KeyError:
                    continue

        # Save full dataset
        output_path = os.path.join(output_dir, "comprehensive_qa_dataset.json")
        self.save_qa_to_json(all_qa, output_path)

        # Save per-topic datasets
        topic_groups = {}
        for qa in all_qa:
            t = qa.get("topic", "general")
            if t not in topic_groups:
                topic_groups[t] = []
            topic_groups[t].append(qa)

        for topic, qas in topic_groups.items():
            topic_path = os.path.join(output_dir, f"qa_{topic}.json")
            self.save_qa_to_json(qas, topic_path)

        return {
            "total_qa": len(all_qa),
            "topics_covered": list(topic_groups.keys()),
            "output_dir": output_dir,
        }


if __name__ == "__main__":
    generator = QAGenerator()
    result = generator.generate_comprehensive_dataset()
    print(f"\nGenerated {result['total_qa']} Q&A pairs across {len(result['topics_covered'])} topics")
    print(f"Output saved to: {result['output_dir']}")