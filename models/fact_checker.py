import re
import urllib.parse
import xml.etree.ElementTree as ET
import requests
import torch
import torch.nn.functional as F

class LiveFactCheckEngine:
    """
    Live Fact-Check Knowledge Grounding Engine.
    Queries verified journalism fact-checking networks (Snopes, India Today, BOOM, AltNews, The Quint, etc.)
    with Option 1: CLIP Semantic Embedding Similarity Filtering to eliminate topic/predicate mismatches.
    """
    FACT_CHECK_DOMAINS = [
        "snopes.com", "indiatoday.in", "boomlive.in", "altnews.in", "thequint.com",
        "newschecker.in", "vishvasnews.com", "factcheck.org", "reuters.com",
        "pib.gov.in", "bbc.com", "timesofindia.indiatimes.com"
    ]

    DEBUNK_KEYWORDS = [
        "deepfake", "doctored", "fake", "false", "misleading",
        "hoax", "fact check", "fabricated", "no!", "untrue", "morphed",
        "myth", "fact or myth", "rumour", "rumor", "debunk", "not visible"
    ]

    STOPWORDS = {
        "a", "an", "the", "in", "on", "at", "by", "for", "with", "about",
        "against", "between", "into", "through", "during", "before", "after",
        "above", "below", "to", "from", "up", "down", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "say",
        "says", "said", "saying", "that", "this", "these", "those", "which", "who",
        "whom", "whose", "what", "where", "when", "why", "how", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
        "should", "now", "they", "them", "their", "theirs", "he", "him", "his", "she"
    }

    def __init__(self, timeout: float = 4.5, feature_extractor=None, sim_threshold: float = 0.72):
        self.timeout = timeout
        self.feature_extractor = feature_extractor
        self.sim_threshold = sim_threshold

    def extract_search_terms(self, text: str) -> str:
        """Extracts prominent named entities and keywords from verbose claims."""
        cleaned = re.sub(r'https?://\S+|www\.\S+', '', text)
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        words = cleaned.split()
        
        filtered = [w for w in words if w.lower() not in self.STOPWORDS and len(w) > 2]
        query_terms = filtered[:7]
        return " ".join(query_terms)

    def verify_claim(self, claim_text: str, feature_extractor=None) -> dict:
        """
        Queries global news and fact-check repositories.
        Enforces Option 1: CLIP Semantic Embedding Cosine Similarity (threshold >= 0.72)
        between user's specific claim and candidate article titles.
        """
        terms = self.extract_search_terms(claim_text)
        if not terms or len(terms.split()) < 2:
            return {"matched": False, "reason": "Insufficient named entities for search."}

        active_extractor = feature_extractor or self.feature_extractor
        claim_emb = None
        if active_extractor is not None:
            try:
                claim_emb = active_extractor.extract_text_features(claim_text)
            except Exception as e:
                print(f"[WARNING] Could not encode claim for semantic fact-checking: {e}")

        # Extract claim core keywords for fallback filtering
        claim_words = set(re.findall(r'\b\w{3,}\b', claim_text.lower())) - self.STOPWORDS

        query = f"{terms} fact check"
        encoded = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        try:
            resp = requests.get(url, headers=headers, timeout=self.timeout)
            if resp.status_code != 200:
                return {"matched": False, "reason": f"HTTP {resp.status_code}"}

            root = ET.fromstring(resp.text)
            items = root.findall('.//item')
            
            fact_hits = []
            
            for item in items:
                title = item.find('title').text if item.find('title') is not None else ""
                link = item.find('link').text if item.find('link') is not None else ""
                source_elem = item.find('source')
                source_name = source_elem.text if source_elem is not None else "News Source"
                
                title_lower = title.lower()
                link_lower = link.lower()
                cleaned_title = re.sub(r' - [^-]+$', '', title).strip()

                # Check if from fact-check network or contains debunk keyword
                is_fact_domain = any(domain in link_lower or domain in title_lower for domain in self.FACT_CHECK_DOMAINS)
                has_debunk_word = any(kw in title_lower for kw in self.DEBUNK_KEYWORDS)
                
                if not (is_fact_domain or has_debunk_word):
                    continue

                # 1. OPTION 1: Deep Semantic Embedding Similarity Gate
                semantic_score = 0.0
                if claim_emb is not None:
                    try:
                        title_emb = active_extractor.extract_text_features(cleaned_title)
                        semantic_score = float(F.cosine_similarity(claim_emb, title_emb).item())
                    except Exception as e:
                        semantic_score = 0.0

                    # Reject candidates that fail the semantic similarity threshold!
                    # E.g. "Snowfall in Sahara" vs "Sahara covered in sand dunes" gives ~0.64 -> REJECTED!
                    if semantic_score < self.sim_threshold:
                        continue
                else:
                    # Fallback when embeddings unavailable: lexical overlap
                    title_words = set(re.findall(r'\b\w{3,}\b', title_lower))
                    overlap = claim_words.intersection(title_words)
                    if len(overlap) < 2:
                        continue

                is_debunked = any(kw in title_lower for kw in [
                    "deepfake", "fake", "false", "doctored", "hoax", "no!", "myth",
                    "fact or myth", "not visible", "untrue", "debunk", "snopes", "misleading"
                ])
                
                fact_hits.append({
                    "title": cleaned_title,
                    "source": source_name,
                    "link": link,
                    "is_debunked": is_debunked,
                    "semantic_similarity": round(semantic_score * 100, 1) if claim_emb is not None else None
                })

            if fact_hits:
                # Sort by debunked status and highest semantic similarity
                fact_hits.sort(key=lambda x: (x["is_debunked"], x.get("semantic_similarity") or 0.0), reverse=True)
                primary = fact_hits[0]
                
                return {
                    "matched": True,
                    "is_debunked": primary["is_debunked"],
                    "confidence_boost": 95.0 if primary["is_debunked"] else 75.0,
                    "primary_hit": primary,
                    "total_sources_found": len(fact_hits),
                    "sources": fact_hits[:3]
                }

        except Exception as e:
            return {"matched": False, "error": str(e)}

        return {"matched": False, "reason": "No semantically matching fact-check articles found."}
