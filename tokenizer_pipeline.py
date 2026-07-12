"""
GPT-2 Style Tokenization Pipeline.
Trains a BPE tokenizer on the extracted corpus and provides
tokenization/encoding utilities for the data loaders.
"""
import os
import json
from typing import List, Dict, Optional, Union
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

from config import TOKENIZED_DIR, EXTRACTED_TEXT_DIR, model_cfg


class GPT2TokenizerPipeline:
    """
    GPT-2 style BPE tokenizer training and encoding pipeline.
    Trains a tokenizer from scratch on the extracted corpus.
    """

    def __init__(self, vocab_size: int = None):
        self.vocab_size = vocab_size or model_cfg.vocab_size
        self.tokenizer_dir = os.path.join(TOKENIZED_DIR, "gpt2_tokenizer")
        os.makedirs(self.tokenizer_dir, exist_ok=True)

        self.tokenizer_path = os.path.join(self.tokenizer_dir, "tokenizer.json")
        self.hf_tokenizer_path = os.path.join(self.tokenizer_dir, "hf_tokenizer")
        self._tokenizer = None
        self._hf_tokenizer = None

    def _get_corpus_files(self) -> List[str]:
        """Get all extracted text files for training."""
        txt_files = []
        if os.path.isdir(EXTRACTED_TEXT_DIR):
            for fname in os.listdir(EXTRACTED_TEXT_DIR):
                if fname.endswith(".txt") and fname != "metadata_index.json":
                    txt_files.append(os.path.join(EXTRACTED_TEXT_DIR, fname))
        return txt_files

    def _create_minimal_corpus(self) -> str:
        """Create a minimal African-focused corpus for tokenizer training."""
        texts = [
            "Africa is a continent of immense diversity with 54 countries.",
            "The African Union promotes unity, peace, and development across the continent.",
            "Democracy and economic growth are key priorities for African nations.",
            "Education and healthcare remain critical challenges requiring investment.",
            "Trade, agriculture, and technology drive innovation in African economies.",
            "South Africa, Nigeria, Kenya, Ghana, and Ethiopia are major economies.",
            "Climate change affects agriculture, water resources, and communities.",
            "Urbanization is rapidly transforming African cities and infrastructure.",
            "Youth populations are growing, creating opportunities and challenges.",
            "Digital transformation is reshaping African economies and societies.",
            "The African Continental Free Trade Area (AfCFTA) creates new markets.",
            "Agenda 2063 envisions an integrated, prosperous, and peaceful Africa.",
            "Renewable energy offers opportunities for sustainable development.",
            "Healthcare systems are being strengthened across the continent.",
            "Infrastructure development connects communities and drives growth.",
            "Good governance and anti-corruption measures strengthen institutions.",
            "Gender equality and women's empowerment advance social progress.",
            "Agricultural innovation improves food security and rural livelihoods.",
            "Cultural heritage and tourism contribute significantly to economic growth.",
            "Peace and security remain essential for sustainable development.",
            "Financial inclusion through mobile banking reaches underserved populations.",
            "Public health initiatives combat malaria, HIV/AIDS, and tuberculosis.",
            "Regional integration through economic communities promotes cooperation.",
            "Natural resource management requires sustainable and equitable approaches.",
            "Small and medium enterprises drive job creation and economic growth.",
            "Climate resilience is essential for vulnerable communities across Africa.",
            "Education technology improves access to quality learning opportunities.",
            "Cross-border trade facilitates regional economic integration and growth.",
            "Water security and sanitation remain critical for public health.",
            "Biodiversity conservation protects Africa's unique natural heritage.",
            "Political participation and civic engagement strengthen democratic processes.",
            "Voting patterns reveal important trends in democratic maturation.",
            "Income inequality varies significantly across different regions of Africa.",
            "Employment trends show a shift toward services and technology sectors.",
            "Migration patterns reflect economic opportunities and environmental factors.",
            "Population growth rates vary widely across the continent's regions.",
            "Family structures are evolving with urbanization and economic changes.",
            "Housing affordability is a growing challenge in rapidly urbanizing areas.",
            "Healthcare access remains uneven between urban and rural populations.",
            "Social mobility through education creates pathways out of poverty.",
            "Agricultural households face challenges from climate variability.",
            "Crime statistics correlate with economic inequality and urbanization.",
            "Community services must adapt to changing demographic patterns.",
            "Cultural diversity enriches African societies and drives innovation.",
            "The African Development Bank finances infrastructure and development projects.",
            "Peace and Security Council of the AU maintains continental stability.",
            "Pan-African Parliament represents the voices of African people.",
            "African Court on Human and Peoples' Rights protects fundamental rights.",
            "Economic Commission for Africa provides research and policy guidance.",
            "African Union Commission coordinates programs across member states.",
            "African Standby Force maintains peace and security in conflict zones.",
            "Africa Centres for Disease Control protects continental health security.",
            "African Space Agency advances space technology for development.",
            "African Monetary Fund promotes financial integration and stability.",
            "African Investment Bank finances transformative infrastructure projects.",
            "African Minerals Development Centre promotes sustainable resource management.",
            "African Union Development Agency (AUDA-NEPAD) implements development programs.",
            "Comprehensive Africa Agriculture Development Programme transforms food systems.",
            "Continental Education Strategy for Africa aims to improve learning outcomes.",
            "Programme for Infrastructure Development in Africa builds connectivity.",
            "Science, Technology and Innovation Strategy for Africa advances research.",
            "African Health Strategy aims to improve healthcare across the continent.",
            "African Governance Architecture strengthens democratic institutions and practices.",
            "African Peace and Security Architecture maintains continental stability.",
            "African Charter on Democracy, Elections and Governance promotes good governance.",
            "African Youth Charter empowers young people across the continent.",
            "African Women's Protocol advances the rights of women and girls.",
            "Silencing the Guns initiative aims to end conflicts by 2030.",
            "African Renaissance promotes cultural revival and continental pride.",
            "Sustainable Development Goals guide Africa's development agenda.",
            "The Africa We Want defines the collective vision for development.",
            "Inclusive growth ensures benefits reach all segments of society.",
            "Human capital development builds skills for the knowledge economy.",
            "Private sector development drives job creation and economic diversification.",
            "Digital government services improve public administration and transparency.",
            "Community-based healthcare reaches remote and underserved populations.",
            "Technology incubators support startup ecosystems and youth entrepreneurship.",
            "Public-private partnerships accelerate infrastructure and service delivery.",
            "Climate adaptation strategies protect vulnerable communities from impacts.",
            "Quality education builds human capital for inclusive growth.",
            "Universal health coverage ensures access to quality care for all.",
            "Sustainable agriculture ensures long-term food security and livelihoods.",
            "Infrastructure corridors connect regional economies and facilitate trade.",
            "Women in leadership drive organizational and societal change.",
            "Youth unemployment remains a critical challenge requiring urgent action.",
            "Environmental conservation protects ecosystems for future generations.",
            "Trade facilitation reduces barriers and promotes cross-border commerce.",
            "Economic diversification reduces dependence on natural resource exports.",
            "Digital transformation of public services improves government efficiency.",
            "Agricultural research develops climate-resilient crop varieties.",
            "Financial literacy empowers individuals and strengthens communities.",
            "Renewable energy investment creates green jobs and sustainable growth.",
            "Gender-responsive budgeting promotes equality in public resource allocation.",
            "Youth leadership development builds capacity for future governance.",
            "Social protection systems reduce vulnerability and build resilience.",
            "Electoral reforms strengthen democratic processes and voter confidence.",
            "Agricultural market access improves incomes for smallholder farmers.",
            "Digital inclusion bridges the technology gap between urban and rural areas.",
            "Financial technology transforms banking and expands access to capital.",
            "Climate finance supports both adaptation and mitigation efforts.",
            "Educational equity ensures access to quality learning for all children.",
            "Healthcare innovation addresses both communicable and non-communicable diseases.",
            "Renewable energy microgrids power rural communities and small businesses.",
            "Trade policy promotes regional economic integration and global competitiveness.",
            "Youth participation strengthens democratic governance and accountability.",
            "Environmental monitoring supports evidence-based conservation decisions.",
            "Social entrepreneurship addresses community needs through innovative models.",
            "Agricultural value chains create employment from farm to market.",
            "Infrastructure investment stimulates economic growth and job creation.",
            "Financial inclusion reaches unbanked populations through digital services.",
            "Educational research informs evidence-based policy and classroom practice.",
            "Healthcare systems strengthening builds resilience against health emergencies.",
            "Digital economy creates new opportunities for African entrepreneurs.",
            "Agricultural cooperatives strengthen farmer bargaining power and livelihoods.",
            "Climate resilience building is essential for food security and stability.",
            "Gender equality programs advance women's economic empowerment and leadership.",
            "Youth skills development prepares the workforce for the digital economy.",
            "Natural resource governance ensures equitable benefit sharing.",
            "Social cohesion strengthens communities and prevents conflict.",
            "Governance transparency builds trust between citizens and institutions.",
            "Agricultural innovation drives productivity gains and food security.",
            "Infrastructure sustainability requires ongoing maintenance and investment.",
            "Financial stability promotes favorable conditions for economic growth.",
            "Educational opportunity creates pathways out of poverty.",
            "Healthcare access reduces health disparities across populations.",
            "Climate action protects current and future generations from impacts.",
            "Trade expansion creates economic opportunities and shared prosperity.",
            "Environmental sustainability ensures resources for future generations.",
            "Social progress requires inclusive policies that benefit all citizens.",
            "Governance effectiveness requires capable institutions and accountable leaders.",
            "Agricultural development reduces rural poverty and improves nutrition.",
            "Infrastructure connectivity links communities to markets and services.",
            "Financial innovation enables inclusive and sustainable economic growth.",
            "Educational excellence transforms individual lives and national development.",
            "Healthcare quality improvement saves lives and improves wellbeing.",
            "Renewable energy transition reduces carbon emissions and powers development.",
            "Trade cooperation builds bridges between nations and regions.",
            "Gender justice advances fundamental human rights and social progress.",
            "Youth opportunity creates pathways for employment and entrepreneurship.",
            "Environmental sustainability protects Africa's natural capital for future generations.",
            "Social justice ensures fairness and equity in development outcomes.",
            "Governance reform strengthens democratic institutions and the rule of law.",
            "Agricultural progress is fundamental to Africa's food sovereignty.",
            "Infrastructure development builds the foundation for tomorrow's growth.",
            "Financial access opens doors to economic opportunity and prosperity.",
            "Educational opportunity transforms individual lives and communities.",
            "Healthcare innovation heals communities and strengthens societies.",
            "Digital future connects everyone to information and opportunity.",
            "Sustainable future ensures prosperity for current and future generations.",
            "Inclusive future ensures that no one is left behind in development.",
            "Peaceful future requires ongoing commitment to dialogue and reconciliation.",
            "Healthy future requires investment in healthcare systems and prevention.",
            "Educated future empowers everyone to reach their full potential.",
            "Connected future links communities across the continent and globally.",
            "Green future protects the environment for coming generations.",
            "Innovative future drives progress through creativity and entrepreneurship.",
            "Resilient future adapts to climate change and other global challenges.",
            "United future builds cooperation across borders for shared prosperity.",
        ]
        return "\n".join(texts)

    def train_tokenizer(self, corpus_files: List[str] = None) -> Tokenizer:
        """
        Train a BPE tokenizer from scratch on the corpus.
        Uses GPT-2 style pre-tokenization (BPE-level).
        """
        if corpus_files is None:
            corpus_files = self._get_corpus_files()

        # Write training corpus to a temp file
        corpus_path = os.path.join(self.tokenizer_dir, "corpus.txt")

        if not corpus_files:
            print("No corpus files found. Using built-in African-focused sample corpus.")
            sample_text = self._create_minimal_corpus()
            with open(corpus_path, "w", encoding="utf-8") as f:
                f.write(sample_text)
            corpus_files = [corpus_path]
        else:
            # Concatenate all corpus files into one for training
            with open(corpus_path, "w", encoding="utf-8") as out:
                for fpath in tqdm(corpus_files, desc="Building corpus"):
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        out.write(f.read())
                        out.write("\n")

        print(f"Training BPE tokenizer with vocab_size={self.vocab_size}...")

        # Initialize GPT-2 style BPE tokenizer
        tokenizer = Tokenizer(models.BPE())

        # GPT-2 uses byte-level pre-tokenization
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        # Train the tokenizer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<|endoftext|>", "<|pad|>"],
            min_frequency=2,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        tokenizer.train(files=[corpus_path], trainer=trainer)

        # Save the tokenizer
        tokenizer.save(self.tokenizer_path)
        print(f"Tokenizer saved to {self.tokenizer_path}")

        # Create HuggingFace-compatible tokenizer
        self._hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
            unk_token="<|endoftext|>",
            model_max_length=model_cfg.max_seq_length,
        )
        self._hf_tokenizer.save_pretrained(self.hf_tokenizer_path)
        print(f"HF Tokenizer saved to {self.hf_tokenizer_path}")

        self._tokenizer = tokenizer
        return tokenizer

    def load_tokenizer(self) -> PreTrainedTokenizerFast:
        """Load a previously trained tokenizer."""
        if os.path.exists(self.hf_tokenizer_path):
            self._hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(
                self.hf_tokenizer_path
            )
            return self._hf_tokenizer
        elif os.path.exists(self.tokenizer_path):
            # Load raw tokenizer and wrap
            raw = Tokenizer.from_file(self.tokenizer_path)
            self._hf_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=raw,
                bos_token="<|endoftext|>",
                eos_token="<|endoftext|>",
                pad_token="<|pad|>",
                unk_token="<|endoftext|>",
                model_max_length=model_cfg.max_seq_length,
            )
            self._hf_tokenizer.save_pretrained(self.hf_tokenizer_path)
            return self._hf_tokenizer
        else:
            raise FileNotFoundError(
                "No trained tokenizer found. Run train_tokenizer() first."
            )

    def encode(self, text: str, add_special_tokens: bool = True) -> Dict:
        """Encode text to token IDs."""
        if self._hf_tokenizer is None:
            self.load_tokenizer()
        encoded = self._hf_tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=model_cfg.max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        if self._hf_tokenizer is None:
            self.load_tokenizer()
        return self._hf_tokenizer.decode(token_ids, skip_special_tokens=True)

    def show_sample_tokenization(self, texts: List[str] = None):
        """Show sample tokenizations to verify the pipeline works."""
        if texts is None:
            texts = [
                "The African Union promotes unity and development across the continent.",
                "South Africa's economy is the most industrialized in Africa.",
                "Climate change poses significant challenges to African agriculture.",
            ]

        if self._hf_tokenizer is None:
            self.load_tokenizer()

        print("\n" + "=" * 80)
        print("SAMPLE TOKENIZATIONS")
        print("=" * 80)
        for i, text in enumerate(texts):
            encoded = self.encode(text)
            decoded = self.decode(encoded["input_ids"])
            print(f"\n--- Sample {i + 1} ---")
            print(f"Original: {text}")
            print(f"Input IDs: {encoded['input_ids'][:20]}...")
            print(f"Attention Mask: {encoded['attention_mask'][:20]}...")
            print(f"Num Tokens: {sum(encoded['attention_mask'])}")
            print(f"Rounded trip decoded: {decoded}")
            print()


if __name__ == "__main__":
    pipeline = GPT2TokenizerPipeline()
    pipeline.train_tokenizer()
    pipeline.show_sample_tokenization()