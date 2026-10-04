import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class MultimodalDataset(Dataset):
    """
    Multimodal Dataset for Fake Content Detection.
    Pairs textual statements with visual images.
    """
    def __init__(self, json_path: str, images_dir: str = "data/images", transform=None):
        self.images_dir = images_dir
        self.transform = transform
        os.makedirs(images_dir, exist_ok=True)

        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                self.samples = json.load(f)
        else:
            self.samples = []

        # Ensure benchmark images exist; generate synthetic visual prototypes if absent
        self._ensure_images()

    def _ensure_images(self):
        """Creates realistic visual test benchmarks for each sample if not present."""
        for item in self.samples:
            img_path = os.path.join(self.images_dir, item["image_source"])
            if not os.path.exists(img_path):
                self._generate_synthetic_sample_image(img_path, item)

    def _generate_synthetic_sample_image(self, path: str, item: dict):
        """Generates representative visual test images with geometric textures and captions."""
        w, h = 512, 512
        category = item.get("category", "authentic")
        
        # Color schemes based on category
        if category == "synthetic_ai":
            # Vibrant neon gradients (diffusion artifacts)
            base = np.zeros((h, w, 3), dtype=np.uint8)
            for y in range(h):
                for x in range(w):
                    base[y, x] = [int((x/w)*255), int((y/h)*150), int(((x+y)/(w+h))*255)]
            img = Image.fromarray(base)
        elif category == "out_of_context":
            # Saturated archival look
            base = np.zeros((h, w, 3), dtype=np.uint8)
            base[:, :] = [70, 90, 110]
            img = Image.fromarray(base)
        else:
            # Natural landscape hue
            base = np.zeros((h, w, 3), dtype=np.uint8)
            base[:, :] = [45, 75, 55]
            img = Image.fromarray(base)

        draw = ImageDraw.Draw(img)
        # Visual cues & borders
        draw.rectangle([10, 10, w - 10, h - 10], outline=(200, 200, 200), width=3)
        draw.text((25, 30), f"Category: {item.get('category', 'Test')}", fill=(255, 255, 255))
        draw.text((25, 60), f"Scene: {item.get('caption', '')[:50]}...", fill=(220, 220, 220))
        
        img.save(path, "JPEG", quality=92)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = os.path.join(self.images_dir, item["image_source"])
        
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        return {
            "id": item["id"],
            "headline": item["headline"],
            "caption": item["caption"],
            "label": torch.tensor(item["label"], dtype=torch.long),
            "label_text": item["label_text"],
            "category": item["category"],
            "image": image,
            "image_path": img_path
        }


def get_dataloaders(json_path: str = "data/sample_dataset.json", batch_size: int = 4, shuffle: bool = True):
    dataset = MultimodalDataset(json_path=json_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x)
    return dataset, loader
