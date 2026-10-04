import io
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import torch

class ImageForensicAnalyzer:
    """
    Performs digital image forensics:
    1. Error Level Analysis (ELA) to expose digital compression & splicing tampering.
    2. Laplacian Noise & High-Frequency variance to detect synthetic diffusion blurring.
    """
    def __init__(self, ela_quality: int = 90, ela_scale: float = 15.0):
        self.ela_quality = ela_quality
        self.ela_scale = ela_scale

    def compute_ela(self, image: Image.Image) -> tuple[Image.Image, float]:
        """
        Calculates Error Level Analysis (ELA).
        Re-saves the image at 90% quality and computes pixel-level absolute difference.
        Returns: (ELA PIL Image, ELA Disparity Metric)
        """
        # Convert to RGB
        rgb_img = image.convert('RGB')
        
        # Save to buffer at fixed quality
        buffer = io.BytesIO()
        rgb_img.save(buffer, 'JPEG', quality=self.ela_quality)
        buffer.seek(0)
        resaved_img = Image.open(buffer)

        # Absolute difference between original and resaved
        diff = ImageChops.difference(rgb_img, resaved_img)

        # Amplify difference for visual human inspection
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema]) if extrema else 1
        scale = self.ela_scale if max_diff == 0 else min(self.ela_scale, 255.0 / max(max_diff, 1))

        enhancer = ImageEnhance.Brightness(diff)
        enhanced_diff = enhancer.enhance(scale)

        # Compute numerical disparity
        diff_arr = np.array(diff, dtype=np.float32)
        mean_disparity = float(np.mean(diff_arr))
        
        return enhanced_diff, mean_disparity

    def analyze_texture_and_noise(self, image: Image.Image) -> dict:
        """
        Computes local variance and gradient sharpness to differentiate
        natural camera sensor grain from AI diffusion smoothness.
        """
        gray = image.convert('L')
        arr = np.array(gray, dtype=np.float32)
        
        # Simple finite-difference Laplacian approximation
        laplacian = (
            -4 * arr[1:-1, 1:-1]
            + arr[:-2, 1:-1] + arr[2:, 1:-1]
            + arr[1:-1, :-2] + arr[1:-1, 2:]
        )
        
        noise_var = float(np.var(laplacian))
        # High noise variance often indicates natural texture or noisy splicing.
        # Unusually low variance indicates artificial smoothing/generation.
        return {
            "noise_variance": noise_var,
            "is_artificially_smooth": noise_var < 80.0
        }

    def analyze(self, image: Image.Image) -> dict:
        """Full image forensic audit."""
        ela_img, ela_disparity = self.compute_ela(image)
        texture_info = self.analyze_texture_and_noise(image)

        # Normalized forensic manipulation score (0 to 100)
        # Typical real images have ELA disparity between 1.5 and 5.0. 
        # Spliced/heavily manipulated images spike above 8.0+.
        manipulation_score = float(np.clip((ela_disparity - 2.0) / 8.0 * 100.0, 5.0, 95.0))
        if texture_info["is_artificially_smooth"]:
            manipulation_score = max(manipulation_score, 65.0)

        # 4-dimensional feature vector for neural fusion
        feat_vec = torch.tensor([
            ela_disparity,
            manipulation_score / 100.0,
            texture_info["noise_variance"] / 1000.0,
            float(texture_info["is_artificially_smooth"])
        ], dtype=torch.float32).unsqueeze(0)

        return {
            "manipulation_score": round(manipulation_score, 2),
            "ela_disparity": round(ela_disparity, 3),
            "noise_variance": round(texture_info["noise_variance"], 2),
            "is_artificially_smooth": texture_info["is_artificially_smooth"],
            "ela_image": ela_img,
            "feature_vector": feat_vec
        }
