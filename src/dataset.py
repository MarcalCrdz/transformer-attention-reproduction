import torch
from torch.utils.data import Dataset, DataLoader

class MassTextDataset(Dataset):
    def __init__(self, num_samples: int, max_len: int):
        self.max_len = max_len
        
        print(f"[*] Gerando {num_samples} amostras massivas localmente na memória...")
        
        # Simulamos 20.000 sequências de IDs numéricos inteiros diretamente (como se já estivessem tokenizados)
        
        self.data_text = torch.randint(2, 7900, (num_samples, max_len), dtype=torch.long)
        
        
        self.labels = torch.cat([torch.zeros(num_samples // 2), torch.ones(num_samples // 2)])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data_text[idx], self.labels[idx]

def get_mass_data_loader(num_samples: int, max_len: int, batch_size: int):
    
    dataset = MassTextDataset(num_samples, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)