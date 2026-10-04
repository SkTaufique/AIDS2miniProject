"""
Multimodal Fake Content Detection Engine
Modules for text analysis, image forensics, CLIP cross-modal alignment, deep fusion, and live fact-checking.
"""

from .clip_encoder import MultimodalFeatureExtractor
from .text_analyzer import TextCredibilityAnalyzer
from .image_forensics import ImageForensicAnalyzer
from .multimodal_classifier import MultimodalFusionClassifier, FusionDecisionEngine
from .explainability import ExplainableAIInspector
from .fact_checker import LiveFactCheckEngine

__all__ = [
    "MultimodalFeatureExtractor",
    "TextCredibilityAnalyzer",
    "ImageForensicAnalyzer",
    "MultimodalFusionClassifier",
    "FusionDecisionEngine",
    "ExplainableAIInspector",
    "LiveFactCheckEngine",
]
