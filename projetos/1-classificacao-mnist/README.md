# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo: Luiz Felipe Miranda de Souza**

### 1️⃣ Resumo da Arquitetura do Modelo

A CNN implementada em "train_model.py" é composta por 3 blocos convolucionais, cada um formado por uma camada Conv2D seguida de BatchNormalization e MaxPooling2D (filtros: 32, 64 e 64, respectivamente, todos com kernel 3x3 e ativação ReLU).
Após os blocos convolucionais, o volume é achatado (Flatten) e passa por uma camada densa de 128 neurônios com ativação ReLU, seguida de uma camada Dropout (taxa de 0.4) antes da camada de saída com 10 neurônios e ativação softmax (uma para cada dígito).

Para o treinamento, foi utilizado um split explícito treino/validação de 90/10 (validation_split = 0.1) sobre o conjunto de treino do MNIST, com EarlyStopping monitorando a val_loss (paciência de 3 épocas e restauração dos melhores pesos).
O treinamento foi limitado a no máximo 15 épocas e executado exclusivamente em CPU.

### 2️⃣ Bibliotecas Utilizadas

- TensorFlow / Keras — 2.21.0
- NumPy

### 3️⃣ Técnica de Otimização do Modelo

Foi utilizada a técnica de Dynamic Range Quantization, aplicada via "tf.lite.TFLiteConverter" com "converter.optimizations = [tf.lite.Optimize.DEFAULT]".
Essa técnica quantiza os pesos do modelo de ponto flutuante (float32) para uma representação de menor precisão, mantendo as ativações em float durante a inferência.
Foi escolhida por não exigir um dataset representativo para calibração, sendo simples de aplicar e eficaz na redução do tamanho do modelo.

### 4️⃣ Resultados Obtidos

- Acurácia de validação final: 99,08%
- Acurácia no conjunto de teste: 99,23%
- Tamanho do "model.h5": 1610,9 KB
- Tamanho do "model.tflite": 140,9 KB
- Redução de tamanho após otimização: 91,3%

### 5️⃣ Comentários Adicionais (Opcional)

Foi um projeto bem completo, dando pra aplicar os conhecimentos adquiridos na capacitação.

### 6️⃣ Exemplo de Inferência

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
