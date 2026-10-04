import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image

from models.clip_encoder import MultimodalFeatureExtractor
from models.text_analyzer import TextCredibilityAnalyzer
from models.image_forensics import ImageForensicAnalyzer
from models.multimodal_classifier import MultimodalFusionClassifier
from data.dataset_loader import MultimodalDataset

def build_feature_tensors(dataset: MultimodalDataset, feature_extractor, text_analyzer, forensic_analyzer):
    """Precomputes frozen multimodal representations for rapid classifier training."""
    print(f"\n[STEP 1] Extracting multimodal features for {len(dataset)} dataset samples...")
    
    feature_list = []
    labels_list = []
    
    for i, item in enumerate(dataset):
        headline = item["headline"]
        image = item["image"]
        label = item["label"].item()
        
        # 1. CLIP Visual & Linguistic Embeddings
        img_feat = feature_extractor.extract_image_features(image) # [1, 512]
        txt_feat = feature_extractor.extract_text_features(headline) # [1, 512]
        sim_data = feature_extractor.compute_similarity(img_feat, txt_feat)
        raw_sim = torch.tensor([[sim_data["raw_cosine"]]], dtype=torch.float32) # [1, 1]
        
        # 2. Textual Cues
        text_data = text_analyzer.analyze(headline)
        text_vec = text_data["feature_vector"] # [1, 6]
        
        # 3. Image Forensics
        forensic_data = forensic_analyzer.analyze(image)
        forensic_vec = forensic_data["feature_vector"] # [1, 4]
        
        # Concatenate 1035-d input vector
        combined_feat = torch.cat([img_feat, txt_feat, raw_sim, text_vec, forensic_vec], dim=-1) # [1, 1035]
        
        feature_list.append(combined_feat.squeeze(0))
        labels_list.append(label)
        
        print(f"  Sample {i+1}/{len(dataset)}: '{headline[:35]}...' -> Sim: {sim_data['semantic_coherence']}% | Label: {label}")

    X = torch.stack(feature_list)
    y = torch.tensor(labels_list, dtype=torch.long)
    return X, y

def train_classifier(epochs: int = 25, batch_size: int = 4, lr: float = 1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f" Multimodal Fake Content Detection - Training Pipeline ")
    print(f" Target Device: {device.upper()}")
    print(f"=======================================================")
    
    # Initialize analyzers
    feature_extractor = MultimodalFeatureExtractor(device=device)
    text_analyzer = TextCredibilityAnalyzer()
    forensic_analyzer = ImageForensicAnalyzer()
    
    # Load dataset
    json_path = os.path.join("data", "sample_dataset.json")
    dataset = MultimodalDataset(json_path=json_path)
    
    X, y = build_feature_tensors(dataset, feature_extractor, text_analyzer, forensic_analyzer)
    
    # Create DataLoader
    train_data = TensorDataset(X, y)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Initialize Multimodal Classifier
    model = MultimodalFusionClassifier(input_dim=1035, hidden_dim=256, num_classes=2, dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"\n[STEP 2] Training Fusion Network for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
        scheduler.step()
        epoch_loss = total_loss / total
        epoch_acc = (correct / total) * 100.0
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:02d}/{epochs}] - Loss: {epoch_loss:.4f} | Training Accuracy: {epoch_acc:.1f}%")

    elapsed = time.time() - start_time
    print(f"\n[STEP 3] Training Complete in {elapsed:.2f} seconds!")
    
    # Final Evaluation & Metrics
    model.eval()
    with torch.no_grad():
        all_x = X.to(device)
        eval_preds = torch.argmax(model(all_x), dim=-1).cpu().numpy()
        true_labels = y.numpy()
        
    acc = np.mean(eval_preds == true_labels) * 100.0
    print(f"\nFinal Evaluation Metrics:")
    print(f"  Overall Accuracy : {acc:.2f}%")
    
    # Save Checkpoint
    checkpoint_dir = os.path.join("models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, "multimodal_best.pt")
    torch.save(model.state_dict(), checkpoint_file)
    print(f"  Model saved to   : {checkpoint_file}")
    print("=======================================================\n")

if __name__ == "__main__":
    train_classifier()
