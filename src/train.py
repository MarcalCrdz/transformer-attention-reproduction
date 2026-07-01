import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_mass_data_loader
from src.model import TransformerClassifier

try:
    import pynvml
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    TELEMETRY_AVAILABLE = True
except Exception:
    TELEMETRY_AVAILABLE = False

def get_vram_usage():
    if TELEMETRY_AVAILABLE:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            return info.used / (1024 ** 2)
        except Exception:
            return 0.0
    return 0.0

def run_mass_experiment():
    
    vocab_size = 8000
    d_model = 128     
    n_heads = 4       
    epochs = 5        
    batch_size = 64   
    max_len = 32      
    lr = 0.001        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Nó de computação ativo: {device}")

    
    train_loader = get_mass_data_loader(num_samples=20000, max_len=max_len, batch_size=batch_size)
    print("[+] DataLoaders locais injetados com sucesso.")

    model = TransformerClassifier(vocab_size, d_model, n_heads).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\n" + "="*50)
    print("      DISPARANDO LOOP DE CARGA MASSIVA (GPU STRESS)      ")
    print("="*50)
    print(f"{'Época':<6} | {'Loss Média':<12} | {'Tempo (s)':<10} | {'VRAM Ativa (MB)':<15}")
    print("-"*55)

    os.makedirs("metrics_output", exist_ok=True)
    
    with open("metrics_output/execution_log.csv", "w") as f:
        f.write("epoch,loss,time_sec,vram_mb\n")

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            epoch_loss = 0.0
            
            model.train()
            for batch_texts, batch_labels in train_loader:
                batch_texts = batch_texts.to(device)
                batch_labels = batch_labels.to(device)

                outputs, _ = model(batch_texts)
                loss = criterion(outputs.squeeze(), batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            elapsed_time = time.time() - start_time
            avg_loss = epoch_loss / len(train_loader)
            vram_mb = get_vram_usage()

            print(f"{epoch:<6} | {avg_loss:<12.4f} | {elapsed_time:<10.4f} | {vram_mb:<15.2f}")
            f.write(f"{epoch},{avg_loss:.4f},{elapsed_time:.4f},{vram_mb:.2f}\n")

if __name__ == "__main__":
    run_mass_experiment()