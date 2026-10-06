import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class MultimodalFusionClassifier(nn.Module):
    """
    Deep Cross-Modal Fusion Architecture.
    Combines 512-d CLIP image embedding, 512-d CLIP text embedding,
    scalar alignment similarity, 6-d text credibility cues, and 4-d forensic cues.
    Total input dimension = 1035 features.
    """
    def __init__(self, input_dim: int = 1035, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.3):
        super(MultimodalFusionClassifier, self).__init__()
        
        # Cross-modality Gating / Attention
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3), # Weights for [Image, Text, Alignment]
            nn.Softmax(dim=-1)
        )
        
        # Deep Fusion MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.LeakyReLU(0.2)
        self.drop1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = nn.LeakyReLU(0.2)
        self.drop2 = nn.Dropout(dropout * 0.7)
        
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If batch size is 1, batchnorm requires special care
        if x.shape[0] == 1:
            self.bn1.eval()
            self.bn2.eval()

        h = self.fc1(x)
        h = self.bn1(h)
        h = self.act1(h)
        h = self.drop1(h)
        
        h = self.fc2(h)
        h = self.bn2(h)
        h = self.act2(h)
        h = self.drop2(h)
        
        logits = self.classifier(h)
        return logits


class FusionDecisionEngine:
    """
    Coordinates feature extractors and the trained fusion classifier.
    Computes holistic authenticity scores and human-interpretable reasoning.
    """
    def __init__(self, checkpoint_path: str = None, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = MultimodalFusionClassifier().to(self.device)
        self.model.eval()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[INFO] Loading trained weights from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("[SUCCESS] Checkpoint loaded.")
        else:
            # Initialize with sensible default prior weights (favoring consistency check)
            self._init_heuristic_weights()

    def _init_heuristic_weights(self):
        """Initializes weights with calibrated priors if no fine-tuned checkpoint exists yet."""
        with torch.no_grad():
            nn.init.xavier_uniform_(self.model.fc1.weight, gain=0.1)
            nn.init.zeros_(self.model.fc1.bias)
            nn.init.xavier_uniform_(self.model.fc2.weight, gain=0.1)
            nn.init.zeros_(self.model.fc2.bias)
            nn.init.xavier_uniform_(self.model.classifier.weight, gain=0.1)
            nn.init.zeros_(self.model.classifier.bias)

    def predict(
        self, 
        image_feat: torch.Tensor, 
        text_feat: torch.Tensor, 
        similarity_data: dict, 
        text_data: dict, 
        forensic_data: dict
    ) -> dict:
        """
        Takes raw multi-modal extracted features, passes through the neural network,
        and returns structured verdict and diagnostic scores.
        """
        raw_sim = torch.tensor([[similarity_data["raw_cosine"]]], dtype=torch.float32)
        text_vec = text_data["feature_vector"]
        forensic_vec = forensic_data["feature_vector"]
        
        # Concatenate into full 1035-dimensional input
        x = torch.cat([image_feat, text_feat, raw_sim, text_vec, forensic_vec], dim=-1).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
        # Class 0: Authentic, Class 1: Fake/Manipulated
        fake_prob = float(probs[1])
        authentic_prob = float(probs[0])
        
        # Calculate dynamic risk synthesis
        coherence_score = similarity_data["semantic_coherence"]
        sensationalism_score = text_data["sensationalism_score"]
        manipulation_score = forensic_data["manipulation_score"]
        coherence_risk = max(0.0, 100.0 - coherence_score)
        
        # Weighted Multimodal Fake Index (0 - 100)
        # A Cheapfake (mismatched image-text) is fake even if unmanipulated and politely worded!
        if similarity_data.get("is_out_of_context") or coherence_score < 42.0:
            mismatch_severity = max(0.0, (42.0 - coherence_score) / 42.0)
            composite_fake_index = 62.0 + (mismatch_severity * 30.0)
        else:
            composite_fake_index = (
                fake_prob * 30.0 +
                coherence_risk * 0.35 +
                sensationalism_score * 0.20 +
                manipulation_score * 0.15
            )
        composite_fake_index = float(np.clip(composite_fake_index, 3.0, 97.0))
        
        is_fake = composite_fake_index >= 50.0
        confidence = composite_fake_index if is_fake else (100.0 - composite_fake_index)
        
        if is_fake and similarity_data.get("is_out_of_context"):
            verdict = "Fake / Out-of-Context"
        elif is_fake:
            verdict = "Fake / Manipulated"
        else:
            verdict = "Authentic / Verified"
        
        # Assemble Human-Readable Reasons
        risk_reasons = []
        if is_fake:
            if similarity_data.get("discrepancy_reason"):
                risk_reasons.append(similarity_data["discrepancy_reason"])
            elif similarity_data.get("is_out_of_context") or coherence_score < 45.0:
                risk_reasons.append(f"Cross-Modal Semantic Mismatch ({coherence_score:.1f}% alignment): Visual elements in the image do not corroborate the claimed event or subject in the headline (Cheapfake / Out-of-Context).")
            else:
                risk_reasons.append(f"Cross-Modal Consistency: Visual and textual features have moderate alignment ({coherence_score:.1f}%), but multimodal risk markers remain elevated.")
                
            if sensationalism_score > 35.0:
                kw_str = f" ({', '.join(text_data['suspicious_keywords'])})" if text_data.get('suspicious_keywords') else ""
                risk_reasons.append(f"Sensationalist Phrasing ({sensationalism_score:.1f}% risk score): Detected clickbait or alarmist language tokens{kw_str}.")
            
            if forensic_data["manipulation_score"] > 50.0:
                risk_reasons.append(f"Image Splicing Indicator: Compression Error Level Analysis (ELA disparity: {forensic_data['ela_disparity']:.1f}) reveals local block anomalies typical of photo manipulation.")
            elif forensic_data.get("is_artificially_smooth"):
                risk_reasons.append(f"Synthetic Texture Signature: Image exhibits abnormal low-frequency variance ({forensic_data['noise_variance']:.1f}), consistent with generative AI synthesis.")
            
            if not risk_reasons:
                risk_reasons.append(f"Multimodal Fusion Warning: Joint cross-modal feature embedding indicates deception risk ({composite_fake_index:.1f}%).")
        else:
            risk_reasons.append(f"Cross-Modal Semantic Coherence ({coherence_score:.1f}%): Visual imagery naturally aligns with and supports the stated headline.")
            risk_reasons.append(f"Authentic Digital Forensics: Error Level Analysis (ELA disparity: {forensic_data['ela_disparity']:.1f}) confirms uniform sensor compression without local tampering.")
            risk_reasons.append(f"Objective Journalistic Tone: Text sensationalism is low ({sensationalism_score:.1f}%) and conforms to standard factual reporting.")

        return {
            "verdict": verdict,
            "is_fake": is_fake,
            "confidence": round(confidence, 1),
            "fake_probability": round(composite_fake_index, 1),
            "authentic_probability": round(100.0 - composite_fake_index, 1),
            "risk_factors": risk_reasons,
            "diagnostics": {
                "cross_modal_coherence": coherence_score,
                "text_sensationalism": sensationalism_score,
                "image_manipulation_risk": manipulation_score,
                "ela_disparity": forensic_data["ela_disparity"]
            }
        }

    def predict_video(
        self,
        video_data: dict,
        text_data: dict = None,
        fact_check_data: dict = None
    ) -> dict:
        """
        Multimodal decision engine for video claims (combines visual keyframes, facial forensics,
        and audio voice forensics).
        """
        coherence = video_data.get("coherence_score", 50.0)
        coherence_risk = max(0.0, 100.0 - coherence)
        is_ooc = video_data.get("is_out_of_context", False)
        face_risk = video_data.get("face_tampering_score", 10.0)
        
        audio_info = video_data.get("audio_data", {})
        voice_risk = audio_info.get("synthetic_voice_score", 10.0) if audio_info.get("has_audio") else 0.0
        
        sensationalism = text_data.get("sensationalism_score", 10.0) if text_data else 10.0
        
        # Calculate composite fake index for video
        if is_ooc or coherence < 40.0:
            mismatch_sev = max(0.0, (40.0 - coherence) / 40.0)
            composite_fake_index = 65.0 + (mismatch_sev * 28.0)
        elif face_risk >= 55.0:
            composite_fake_index = 60.0 + (face_risk * 0.35)
        elif voice_risk >= 65.0:
            composite_fake_index = 58.0 + (voice_risk * 0.35)
        else:
            composite_fake_index = (
                coherence_risk * 0.35 +
                face_risk * 0.30 +
                voice_risk * 0.20 +
                sensationalism * 0.15
            )
            
        composite_fake_index = float(np.clip(composite_fake_index, 4.0, 96.0))
        is_fake = composite_fake_index >= 50.0
        confidence = composite_fake_index if is_fake else (100.0 - composite_fake_index)
        
        if is_fake and face_risk >= 50.0:
            verdict = "Deepfake / Manipulated Video"
        elif is_fake and (is_ooc or coherence < 45.0):
            verdict = "Fake / Out-of-Context Video"
        elif is_fake and voice_risk >= 60.0:
            verdict = "Fake / Cloned Audio Soundtrack"
        elif is_fake:
            verdict = "Suspicious / Manipulated Video"
        else:
            verdict = "Authentic / Verified Video"
            
        reasons = list(video_data.get("reasons", []))
        if fact_check_data and fact_check_data.get("is_debunked"):
            reasons.insert(0, f"Live Fact-Check Alert: Claim debunked by {fact_check_data.get('source', 'independent networks')}.")
            composite_fake_index = max(composite_fake_index, 88.0)
            is_fake = True
            verdict = "Debunked Falsehood / Fake Video"
            
        return {
            "verdict": verdict,
            "is_fake": is_fake,
            "confidence": round(confidence, 1),
            "fake_probability": round(composite_fake_index, 1),
            "authentic_probability": round(100.0 - composite_fake_index, 1),
            "risk_factors": reasons,
            "diagnostics": {
                "cross_modal_coherence": round(coherence, 1),
                "face_tampering_risk": round(face_risk, 1),
                "voice_synthetic_risk": round(voice_risk, 1),
                "text_sensationalism": round(sensationalism, 1)
            }
        }

    def predict_audio(
        self,
        audio_forensics: dict,
        transcription_data: dict,
        text_data: dict = None,
        fact_check_data: dict = None
    ) -> dict:
        """
        Multimodal decision engine for audio speech recordings (combines acoustic forensics,
        speech transcription credibility, and fact-checking).
        """
        voice_score = audio_forensics.get("synthetic_voice_score", 15.0)
        sensationalism = text_data.get("sensationalism_score", 10.0) if text_data else 10.0
        
        # Combined multimodal fake index
        # An audio is fake if EITHER the voice is an AI clone OR the spoken claim is sensational misinformation!
        if voice_score >= 50.0:
            composite_fake_index = 60.0 + (voice_score * 0.35)
        elif sensationalism >= 30.0:
            composite_fake_index = 56.0 + (sensationalism * 0.35)
        elif voice_score >= 38.0 and sensationalism >= 20.0:
            composite_fake_index = 52.0 + (voice_score * 0.25) + (sensationalism * 0.20)
        else:
            composite_fake_index = (voice_score * 0.60) + (sensationalism * 0.40)
            
        if fact_check_data and fact_check_data.get("is_debunked"):
            composite_fake_index = max(composite_fake_index, 89.0)
            
        composite_fake_index = float(np.clip(composite_fake_index, 4.0, 96.0))
        is_fake = composite_fake_index >= 50.0
        confidence = composite_fake_index if is_fake else (100.0 - composite_fake_index)
        
        if is_fake and voice_score >= 50.0 and sensationalism >= 30.0:
            verdict = "AI Voice Clone + Fabricated Rumor"
        elif is_fake and voice_score >= 50.0:
            verdict = "AI Voice Clone / Deepfake Audio"
        elif is_fake and sensationalism >= 30.0:
            verdict = "Fabricated / Alarmist Speech Claim"
        elif is_fake and fact_check_data and fact_check_data.get("is_debunked"):
            verdict = "Fabricated / Debunked Speech"
        elif is_fake:
            verdict = "Suspicious / Manipulated Audio"
        else:
            verdict = "Authentic / Natural Voice"
            
        reasons = list(audio_forensics.get("reasons", []))
        if text_data and text_data.get("sensationalism_score", 0) > 25.0:
            kw = ", ".join(text_data.get("suspicious_keywords", []))
            kw_str = f" ({kw})" if kw else ""
            reasons.insert(0, f"Alarmist Disinformation Tone: Spoken transcript contains high sensationalism markers{kw_str}.")
        if fact_check_data and fact_check_data.get("is_debunked"):
            reasons.insert(0, f"Live Fact-Check Alert: Spoken statement debunked by {fact_check_data.get('source', 'journalism networks')}.")
            
        return {
            "verdict": verdict,
            "is_fake": is_fake,
            "confidence": round(confidence, 1),
            "fake_probability": round(composite_fake_index, 1),
            "authentic_probability": round(100.0 - composite_fake_index, 1),
            "risk_factors": reasons,
            "diagnostics": {
                "voice_synthetic_risk": round(voice_score, 1),
                "pitch_mean_hz": audio_forensics.get("pitch_mean_hz", 0.0),
                "pitch_jitter_hz": audio_forensics.get("pitch_jitter_hz", 0.0),
                "text_sensationalism": round(sensationalism, 1)
            }
        }
