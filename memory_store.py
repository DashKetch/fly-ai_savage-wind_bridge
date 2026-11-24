import os
import torch
import pickle
from pathlib import Path

class MemoryPersistence:
    def __init__(self, memory_dir="botw_memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.buffer_path = self.memory_dir / "experience_buffer.pkl"
        self.weights_path = self.memory_dir / "readout_weights.pt"

    def save_state(self, experience_buffer, model_weights):
        # Save experience buffer
        with open(self.buffer_path, "wb") as f:
            pickle.dump(experience_buffer, f)
            
        # Save readout weights
        torch.save(model_weights, self.weights_path)
        print(f"[MemoryPersistence] Successfully persisted state to {self.memory_dir}/")

    def load_state(self):
        buffer, weights = None, None
        
        if self.buffer_path.exists():
            with open(self.buffer_path, "rb") as f:
                buffer = pickle.load(f)
            print(f"[MemoryPersistence] Loaded experience buffer ({len(buffer)} transitions).")
            
        if self.weights_path.exists():
            weights = torch.load(self.weights_path)
            print("[MemoryPersistence] Loaded neural readout weights.")
            
        return buffer, weights