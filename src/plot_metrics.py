import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_report_plots():
    csv_path = "metrics_output/execution_log.csv"
    if not os.path.exists(csv_path):
        print(f"[-] Erro: {csv_path} não encontrado. Rode o treino primeiro.")
        return

    
    df = pd.read_csv(csv_path)

    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    color_accent = '#00FF66'  
    color_vram = '#00CCFF'    
    
    ax1.plot(df['epoch'], df['loss'], color=color_accent, linewidth=2, marker='o', markersize=4)
    ax1.set_title("CURVA DE CONVERGÊNCIA DA PERDA (LOSS)", fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlabel("Época", fontsize=9)
    ax1.set_ylabel("Loss Valor", fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.3)

    
    ax2.plot(df['epoch'], df['vram_mb'], color=color_vram, linewidth=2, marker='s', markersize=4)
    ax2.set_title("ALOCAÇÃO DE MEMÓRIA VRAM (RTX 2060)", fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel("Época", fontsize=9)
    ax2.set_ylabel("Uso de VRAM (MB)", fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    
   
    output_img = "metrics_output/performance_metrics.png"
    plt.savefig(output_img, dpi=300)
    print(f"[+] Gráficos de telemetria gerados com sucesso em: {output_img}")

if __name__ == "__main__":
    generate_report_plots()