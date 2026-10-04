import re
import numpy as np
import torch

class TextCredibilityAnalyzer:
    """
    Evaluates linguistic cues, sensationalism, clickbait markers,
    and emotional exaggeration typical of deceptive multimodal posts.
    """
    SENSATIONAL_PATTERNS = [
        r'\b(shocking|unbelievable|secret|hidden|exposed|truth|conspiracy|alert|urgent)\b',
        r'\b(you won\'t believe|they don\'t want you to know|must watch|viral|breaking)\b',
        r'\b(miracle|cure|scam|warning|danger|leaked|banned|proof|alien)\b',
        r'\b(disaster|horrifying|terrifying|catastrophe|apocalypse)\b'
    ]

    def __init__(self):
        self.regexes = [re.compile(p, re.IGNORECASE) for p in self.SENSATIONAL_PATTERNS]

    def analyze(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                "sensationalism_score": 0.0,
                "clickbait_risk": "Low",
                "uppercase_ratio": 0.0,
                "punctuation_intensity": 0.0,
                "suspicious_keywords": [],
                "feature_vector": torch.zeros(1, 6)
            }

        cleaned = text.strip()
        words = cleaned.split()
        total_words = max(len(words), 1)

        # 1. Sensational & Clickbait keywords match
        detected_keywords = []
        for reg in self.regexes:
            matches = reg.findall(cleaned)
            if matches:
                detected_keywords.extend(matches)
        
        sensational_count = len(detected_keywords)
        sensational_ratio = min(sensational_count / (total_words * 0.2 + 1), 1.0)

        # 2. Uppercase ratio (excessive capitalization often signals ragebait / spam)
        letters = [c for c in cleaned if c.isalpha()]
        uppercase_ratio = sum(1 for c in letters if c.isupper()) / max(len(letters), 1)
        uppercase_penalty = min(uppercase_ratio * 1.5, 1.0) if uppercase_ratio > 0.3 else 0.0

        # 3. Punctuation intensity (e.g. "!!!", "???", "?!")
        multi_punct = len(re.findall(r'[!?]{2,}', cleaned))
        punct_score = min(multi_punct * 0.25, 1.0)

        # 4. Exaggeration / Urgency density
        urgency_markers = len(re.findall(r'\b(now|today|hurry|share before deleted|spread)\b', cleaned, re.IGNORECASE))
        urgency_score = min(urgency_markers * 0.3, 1.0)

        # Composite Sensationalism Score (0 to 100%)
        composite_score = (
            sensational_ratio * 0.40 +
            uppercase_penalty * 0.25 +
            punct_score * 0.20 +
            urgency_score * 0.15
        ) * 100.0

        composite_score = float(np.clip(composite_score, 0.0, 100.0))

        if composite_score > 60:
            risk_level = "High"
        elif composite_score > 25:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # 6-dimensional feature vector for multimodal neural fusion
        feat_vec = torch.tensor([
            sensational_ratio,
            uppercase_ratio,
            punct_score,
            urgency_score,
            composite_score / 100.0,
            float(len(cleaned) > 280) # Long form vs microblog
        ], dtype=torch.float32).unsqueeze(0)

        return {
            "sensationalism_score": round(composite_score, 2),
            "clickbait_risk": risk_level,
            "uppercase_ratio": round(uppercase_ratio * 100, 1),
            "punctuation_intensity": round(punct_score * 100, 1),
            "suspicious_keywords": list(set(detected_keywords))[:5],
            "feature_vector": feat_vec
        }
