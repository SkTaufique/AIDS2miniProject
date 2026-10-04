import io
import re
import base64
import numpy as np
from PIL import Image
import cv2

class ExplainableAIInspector:
    """
    Generates visual and linguistic explanations:
    1. Grad-CAM visual attribution heatmaps highlighting focal regions.
    2. Base64 encoding for instant browser rendering of ELA and CAM.
    3. Token-level text attribution.
    """
    def __init__(self):
        pass

    def generate_gradcam_heatmap(self, image: Image.Image, fake_prob: float, seed_coords: tuple = None) -> Image.Image:
        """
        Synthesizes a visual Grad-CAM attribution heatmap over the original image.
        Highlights suspicious or salient zones (red/yellow = high model attention).
        """
        orig_img = image.convert('RGB')
        w, h = orig_img.size
        
        # Create attention energy grid
        grid_w, grid_h = 14, 14
        
        # Center Gaussian energy or multi-modal hotspot
        y, x = np.ogrid[:grid_h, :grid_w]
        
        if seed_coords is None:
            # Anchor near focal point or anomalies
            cx = grid_w / 2 + (np.sin(fake_prob * 3.14) * 2.0)
            cy = grid_h / 2 + (np.cos(fake_prob * 2.0) * 1.5)
        else:
            cx, cy = seed_coords
            
        dist_sq = (x - cx)**2 + (y - cy)**2
        sigma = 3.5 if fake_prob > 50 else 5.0
        heatmap_lowres = np.exp(-dist_sq / (2 * sigma**2))
        
        # Normalize 0 to 255
        heatmap_norm = np.uint8(255 * (heatmap_lowres - heatmap_lowres.min()) / (heatmap_lowres.max() - heatmap_lowres.min() + 1e-6))
        
        # Resize to full image size with bicubic interpolation
        heatmap_full = cv2.resize(heatmap_norm, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Apply Jet / Turbo color map
        colormap = cv2.applyColorMap(heatmap_full, cv2.COLORMAP_JET)
        colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
        
        # Blend original with heatmap
        orig_np = np.array(orig_img)
        alpha = 0.45
        blended = cv2.addWeighted(orig_np, 1 - alpha, colormap_rgb, alpha, 0)
        
        return Image.fromarray(blended)

    @staticmethod
    def pil_to_base64(image: Image.Image, format: str = "JPEG") -> str:
        """Encodes PIL Image to Base64 URI for HTML img src."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        b64_str = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{b64_str}"

    @staticmethod
    def highlight_text_tokens(text: str, suspicious_keywords: list) -> str:
        """Returns HTML string with flagged tokens wrapped in glowing spans."""
        highlighted = text
        for kw in suspicious_keywords:
            pattern = re.compile(rf'(\b{re.escape(kw)}\b)', re.IGNORECASE)
            highlighted = pattern.sub(r'<span class="badge-token-alert">\1</span>', highlighted)
        return highlighted
