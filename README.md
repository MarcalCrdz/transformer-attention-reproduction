Reprodução Estatística e de Infraestrutura do Mecanismo de Multi-Head Attention

Este repositório contém a implementação do zero (from scratch) do mecanismo de Multi-Head Attention, o componente central da arquitetura Transformer descrita no artigo científico original "Attention Is All You Need" (Vaswani et al., 2017).

O diferencial deste projeto é o foco prático em Engenharia de Sistemas e Infraestrutura, monitorando o consumo de memória de vídeo (VRAM) e a estabilidade de processamento em tempo real diretamente em hardware de consumo local (NVIDIA RTX 2060).

🎯 Objetivos do Projeto

Isolamento Matemático: Implementar as matrizes de Projeção Linear ($Q, K, V$) e o cálculo de produto escalar escalado (Scaled Dot-Product) sem depender de camadas pré-prontas de bibliotecas de alto nível.

Pipeline de Carga Estável: Criar um gerador de dados de alta densidade em memória RAM para testar, estressar e validar a convergência do algoritmo de atenção.

Telemetria de Hardware: Capturar as métricas reais de desempenho e alocação de memória da GPU utilizando a biblioteca de diagnóstico oficial da NVIDIA (pynvml).

🏗️ Estrutura do Projeto

O código está organizado seguindo padrões de design modular de software de Machine Learning:

transformer-attention-reproduction/
├── metrics_output/
│ ├── execution_log.csv # Histórico de Loss e VRAM coletado por época
│ └── performance_metrics.png # Gráfico gerado de telemetria e convergência
├── src/
│ ├── **init**.py
│ ├── attention.py # Algoritmo matemático do Scaled Dot-Product Attention
│ ├── dataset.py # Criação e carregamento do dataset massivo local
│ ├── model.py # Integração da arquitetura Multi-Head e classificador
│ ├── plot_metrics.py # Script visualizador de métricas com Matplotlib
│ └── predict.py # Script de teste de inferência e interpretabilidade
├── .gitignore
└── README.md

📊 Resultados e Telemetria de Hardware

O modelo foi validado sob estresse com um volume de 20.000 amostras e tamanho de lote de 64 (batch size).

🔬 Análise Técnica do Experimento

Convergência Gradual: A curva de perda (Loss Média) decaiu de forma consistente e suave ao longo das épocas ($0.6942 \to 0.4363$), validando que o fluxo de retropropagação (backpropagation) calculou corretamente os gradientes sobre o mecanismo de atenção.

Otimização de VRAM: No início do processamento (warmup), a GPU registrou uma alocação de 1677.78 MB. A partir da 4ª época, o coletor de lixo dinâmico do PyTorch otimizou o grafo computacional, estabilizando o consumo em 1588.93 MB.

Estabilização de Vazão (Throughput): A primeira época consumiu 1.94 segundos devido ao tempo de compilação dos kernels CUDA. O tempo de processamento das épocas seguintes estabilizou em 1.20 segundos.

⚡ Como Executar o Projeto

1. Clonar o repositório e ativar o ambiente virtual:

git clone [https://github.com/MarcalCrdz/transformer-attention-reproduction](https://github.com/MarcalCrdz/transformer-attention-reproduction)
cd transformer-attention-reproduction
python -m venv venv
source venv/bin/activate # No Linux/Mac
.\venv\Scripts\Activate.ps1 # No Windows (PowerShell)

2. Instalar as dependências necessárias:

pip install torch matplotlib pandas pynvml

3. Rodar o pipeline de treinamento e monitoramento:

python -m src.train

4. Gerar os gráficos de performance:

python -m src.plot_metrics

5. Executar uma inferência individual com diagnóstico explicativo:

python -m src.predict

🔮 Interpretabilidade do Modelo

O script de inferência permite visualizar a estrutura de decisão do Transformer:

Formato da Matriz de Atenção Extraída: torch.Size([1, 4, 32, 32])

1: Tamanho do lote (inferência individual).

4: Quantidade de cabeças de atenção (heads) analisando relações contextuais diferentes ao mesmo tempo.

32 x 32: Matriz de afinidade, representando a força de conexão de cada termo com todas as outras palavras que compõem a sequência.
