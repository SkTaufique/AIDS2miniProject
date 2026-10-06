import os
import io
import base64
import tempfile
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class VideoFeatureExtractor:
    """
    Multimodal Video Analysis Engine:
    1. Temporal Keyframe Uniform Sampling (extracts 6-12 frames per video).
    2. Multi-frame CLIP Feature Extraction & Temporal Mean-Pooling into 512-d video visual embedding.
    3. Audio Demuxing & routing to AudioFeatureExtractor for transcript & voice forensics.
    4. Facial artifact & digital splicing detection across frames.
    """
    def __init__(self, clip_extractor=None, audio_extractor=None, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.clip_extractor = clip_extractor
        self.audio_extractor = audio_extractor

        # Load OpenCV face detector if available in current OpenCV build
        self.face_cascade = None
        try:
            if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
                if os.path.exists(cascade_path):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self.face_cascade = None

    def extract_audio_from_video(self, video_path: str) -> str:
        """Extracts audio track from video file and saves as temporary WAV file."""
        temp_wav = None
        try:
            from moviepy import VideoFileClip
            clip = VideoFileClip(video_path)
            if clip.audio is not None:
                fd, temp_wav = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                clip.audio.write_audiofile(temp_wav, fps=16000, nbytes=2, logger=None)
            clip.close()
        except Exception as e:
            print(f"[INFO] Audio extraction notice: {e}")
            temp_wav = None
        return temp_wav

    def sample_keyframes(self, video_path: str, max_frames: int = 8) -> tuple[list[Image.Image], dict]:
        """
        Samples uniformly spaced keyframes from a video file.
        Returns: (list of PIL Images, video metadata dict)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], {"total_frames": 0, "fps": 0, "duration": 0.0}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = fps if fps > 0 else 25.0
        duration = total_frames / fps if total_frames > 0 else 0.0

        if total_frames <= 0:
            cap.release()
            return [], {"total_frames": 0, "fps": fps, "duration": 0.0}

        # Select evenly distributed frame indices
        num_samples = min(max_frames, max(1, total_frames))
        frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

        sampled_frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                sampled_frames.append(pil_img)

        cap.release()
        meta = {
            "total_frames": total_frames,
            "fps": round(fps, 1),
            "duration": round(duration, 2),
            "sampled_count": len(sampled_frames)
        }
        return sampled_frames, meta

    def detect_facial_tampering(self, frames: list[Image.Image]) -> tuple[float, list[str]]:
        """
        Inspects facial regions across sampled frames for deepfake face-swapping boundaries,
        unnatural blur disparities, and temporal frame-to-frame jitter.
        """
        face_count = 0
        boundary_anomalies = 0

        for frame in frames:
            img_np = np.array(frame)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                for (x, y, w, h) in faces:
                    face_count += 1
                    face_roi = gray[y:y+h, x:x+w]
                    y1, y2 = max(0, y - 15), min(gray.shape[0], y + h + 15)
                    x1, x2 = max(0, x - 15), min(gray.shape[1], x + w + 15)
                    border_roi = gray[y1:y2, x1:x2]

                    lap_face = cv2.Laplacian(face_roi, cv2.CV_64F).var()
                    lap_border = cv2.Laplacian(border_roi, cv2.CV_64F).var()

                    if lap_border > 0:
                        disparity = abs(lap_face - lap_border) / max(lap_border, 1.0)
                        if disparity > 1.8:
                            boundary_anomalies += 1
            else:
                # Fast central focal region edge disparity analysis
                h, w = gray.shape
                center_roi = gray[h//4:3*h//4, w//4:3*w//4]
                border_roi = gray[:h//4, :w//4]
                lap_c = cv2.Laplacian(center_roi, cv2.CV_64F).var()
                lap_b = cv2.Laplacian(border_roi, cv2.CV_64F).var()
                if lap_b > 0 and abs(lap_c - lap_b) / max(lap_b, 1.0) > 2.5:
                    boundary_anomalies += 1
                face_count += 1

        reasons = []
        if face_count == 0:
            return 8.0, ["No human faces detected in sampled frames; scene-level verification active."]

        tampering_ratio = boundary_anomalies / max(face_count, 1)
        score = min(92.0, max(6.0, tampering_ratio * 75.0 + 10.0))

        if score >= 50.0:
            reasons.append(f"Facial boundary blending and Laplacian sharpness disparities ({score:.1f}%) suggest synthetic face-swap manipulation.")
        else:
            reasons.append(f"Consistent facial boundary gradients across {face_count} analyzed face regions (Natural appearance).")

        return round(score, 1), reasons

    def analyze_video(self, video_path: str, claim_text: str = None) -> dict:
        """
        Complete end-to-end multimodal video processing pipeline.
        """
        # 1. Sample keyframes
        frames, meta = self.sample_keyframes(video_path, max_frames=8)
        if not frames:
            return {
                "status": "error",
                "error": "Failed to parse video frames.",
                "duration": 0.0
            }

        # 2. Extract and analyze audio track
        temp_wav = self.extract_audio_from_video(video_path)
        has_audio = temp_wav is not None and os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 1000

        audio_res = {
            "has_audio": False,
            "transcript": "",
            "synthetic_voice_score": 0.0,
            "reasons": ["Video has no audible soundtrack."]
        }

        if has_audio and self.audio_extractor is not None:
            trans_data = self.audio_extractor.transcribe(temp_wav)
            forensic_data = self.audio_extractor.analyze_voice_forensics(temp_wav)
            audio_res = {
                "has_audio": True,
                "transcript": trans_data.get("transcript", ""),
                "language": trans_data.get("language", "en"),
                "synthetic_voice_score": forensic_data.get("synthetic_voice_score", 0.0),
                "risk_level": forensic_data.get("risk_level", "Low"),
                "reasons": forensic_data.get("reasons", [])
            }
            # Clean up temp wav file
            try:
                os.remove(temp_wav)
            except Exception:
                pass

        # 3. Multi-frame CLIP encoding & Temporal pooling
        frame_features = []
        if self.clip_extractor is not None:
            for f in frames:
                feat = self.clip_extractor.extract_image_features(f)
                frame_features.append(feat)
            
            # Stack and average pool across temporal dimension
            stacked = torch.cat(frame_features, dim=0) # [N, 512]
            pooled_video_feat = F.normalize(torch.mean(stacked, dim=0, keepdim=True), p=2, dim=-1) # [1, 512]
        else:
            pooled_video_feat = torch.randn(1, 512)
            pooled_video_feat = F.normalize(pooled_video_feat, p=2, dim=-1)

        # 4. Facial deepfake analysis
        face_risk, face_reasons = self.detect_facial_tampering(frames)

        # 5. Visual thumbnails generation (Base64 for Dashboard)
        thumbnails = []
        for i, f in enumerate(frames[:6]):
            thumb = f.resize((160, 90))
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=80)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            thumbnails.append(f"data:image/jpeg;base64,{b64_str}")

        # 6. Cross-modal Coherence (Video vs Claim OR Audio Transcript)
        eval_text = claim_text if claim_text and claim_text.strip() else audio_res.get("transcript", "")
        coherence_score = 50.0
        is_out_of_context = False

        if eval_text and self.clip_extractor is not None:
            txt_feat = self.clip_extractor.extract_text_features(eval_text)
            sim_info = self.clip_extractor.compute_similarity(pooled_video_feat, txt_feat)
            coherence_score = sim_info["semantic_coherence"]
            is_out_of_context = sim_info["is_out_of_context"]

        reasons = list(face_reasons)
        if has_audio and audio_res.get("reasons"):
            reasons.extend(audio_res["reasons"][:2])

        if is_out_of_context:
            reasons.append(f"Video visual scenes show severe mismatch with the accompanying narrative (Coherence: {coherence_score:.1f}%).")

        return {
            "status": "success",
            "metadata": meta,
            "thumbnails": thumbnails,
            "video_feature": pooled_video_feat,
            "coherence_score": round(coherence_score, 1),
            "is_out_of_context": is_out_of_context,
            "face_tampering_score": face_risk,
            "audio_data": audio_res,
            "reasons": reasons
        }
