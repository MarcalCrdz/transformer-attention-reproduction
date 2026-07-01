import torch
import torch.nn as nn
from src.model import TransformerClassifier

def predict_and_explain(text_prompt: str):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 8000
    d_model = 128
    n_heads = 4
    max_len = 32

    
    model = TransformerClassifier(vocab_size, d_model, n_heads).to(device)
    model.eval()  

    
    mock_vocab = {"falha": 10, "crítica": 11, "sistema": 12, "operando": 13, "normal": 14}
    
    words = text_prompt.lower().split()
    token_ids = [mock_vocab.get(w, 42) for w in words] 
    
    
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    else:
        token_ids = token_ids + [0] * (max_len - len(token_ids))

    
    input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

    
    with torch.no_grad():
        logits, attention_weights = model(input_tensor)
        
        
        probability = torch.sigmoid(logits).item()

    
    print("\n" + "="*50)
    print("         DIAGNÓSTICO DE INFERÊNCIA INDIVIDUAL         ")
    print("="*50)
    print(f"Texto Avaliado: \"{text_prompt}\"")
    print(f"Probabilidade de ser Anomalia/Falha: {probability * 100:.2f}%")
    
    
    status = "ALERTA: ANOMALIA DETECTADA" if probability >= 0.5 else "NOMINAL: SISTEMA ESTÁVEL"
    print(f"Classificação Final: {status}")
    print("-" * 50)
    print(f"Shape da Matriz de Atenção Extraída: {attention_weights.shape}")
    print("[*] Nota: A matriz acima descreve a correlação matemática")
    print("    que cada cabeça (head) calculou entre as palavras.")

if __name__ == "__main__":
    
    predict_and_explain("Falha crítica detectada no sistema")