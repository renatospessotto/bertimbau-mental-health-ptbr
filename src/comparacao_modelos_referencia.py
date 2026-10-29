# =============================================================================
# CÉLULA: Comparação Pysentimiento e XLM-RoBERTa vs. rótulos humanos (300 amostras)
#
# PRÉ-REQUISITO: rodar as células anteriores do notebook até a célula
#   "1.5.1 - Método I: Holdout" (inclusive), de modo que a variável
#   data_splits_holdout['teste'] já exista na sessão.
#
# O que esta célula faz:
#   1. Roda Pysentimiento no conjunto de teste humano (300 amostras)
#   2. Roda XLM-RoBERTa no mesmo conjunto
#   3. Mapeia POS/NEU/NEG → B/N/M (rótulos humanos)
#   4. Calcula accuracy, F1-macro, F1-ponderado e relatório por classe
#      para cada modelo de referência, usando os rótulos humanos como GT
#   5. Exibe tabela comparativa: Pysentimiento | XLM-RoBERTa | BERTimbau
#   6. Plota matrizes de confusão lado a lado
# =============================================================================

# ── 0. Dependências ──────────────────────────────────────────────────────────
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# Instala pysentimiento se necessário
try:
    from pysentimiento import create_analyzer
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "pysentimiento", "-q"])
    from pysentimiento import create_analyzer

from transformers import pipeline

# ── 1. Conjunto de teste com rótulos humanos ─────────────────────────────────
df_teste = data_splits_holdout['teste'].copy().reset_index(drop=True)
textos   = df_teste['comentario'].tolist()
y_true   = df_teste['rotulo'].tolist()           # rótulos humanos: B / N / M

print(f"Conjunto de teste: {len(df_teste)} amostras")
print("Distribuição:", df_teste['rotulo'].value_counts().to_dict())

# ── 2. Mapeamento de rótulos ──────────────────────────────────────────────────
# Modelos de referência usam POS / NEU / NEG
# Nosso esquema humano usa  B   / N   / M
MODEL_TO_HUMAN = {"POS": "B", "NEU": "N", "NEG": "M"}
HUMAN_TO_MODEL = {"B": "POS", "N": "NEU", "M": "NEG"}

# ground truth no espaço dos modelos (para o relatório por classe)
y_true_model = [HUMAN_TO_MODEL[r] for r in y_true]

# ── 3. Pysentimiento ──────────────────────────────────────────────────────────
print("\n[1/2] Rodando Pysentimiento nos 300 comentários de teste...")

device = 0 if torch.cuda.is_available() else -1
analyzer_pys = create_analyzer(task="sentiment", lang="pt")

pys_preds = []
BATCH = 64
for i in tqdm(range(0, len(textos), BATCH), desc="Pysentimiento"):
    batch = textos[i : i + BATCH]
    results = analyzer_pys.predict(batch)
    pys_preds.extend([r.output for r in results])   # POS / NEU / NEG

# Mapeia para rótulos humanos
pys_human = [MODEL_TO_HUMAN[p] for p in pys_preds]

# Limpa memória
del analyzer_pys
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ── 4. XLM-RoBERTa ───────────────────────────────────────────────────────────
print("\n[2/2] Rodando XLM-RoBERTa nos 300 comentários de teste...")

xlm_pipe = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    device=device,
    truncation=True,
    max_length=512,
)

# Mapeamento de labels do modelo Cardiff → POS/NEU/NEG
CARDIFF_MAP = {"positive": "POS", "neutral": "NEU", "negative": "NEG"}

xlm_raw = []
for i in tqdm(range(0, len(textos), BATCH), desc="XLM-RoBERTa"):
    batch = textos[i : i + BATCH]
    out   = xlm_pipe(batch)
    xlm_raw.extend([CARDIFF_MAP[r["label"].lower()] for r in out])

xlm_human = [MODEL_TO_HUMAN[p] for p in xlm_raw]

del xlm_pipe
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ── 5. Métricas ───────────────────────────────────────────────────────────────
LABELS_H = ["B", "N", "M"]

def metricas(y_true_h, y_pred_h, nome):
    acc = accuracy_score(y_true_h, y_pred_h)
    f1m = f1_score(y_true_h, y_pred_h, average="macro",    labels=LABELS_H, zero_division=0)
    f1w = f1_score(y_true_h, y_pred_h, average="weighted", labels=LABELS_H, zero_division=0)
    print(f"\n{'═'*60}")
    print(f"  {nome}")
    print(f"{'═'*60}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1-macro   : {f1m:.4f}")
    print(f"  F1-weighted: {f1w:.4f}")
    print()
    print(classification_report(y_true_h, y_pred_h, labels=LABELS_H, zero_division=0))
    return {"Accuracy": acc, "F1-macro": f1m, "F1-ponderado": f1w}

m_pys = metricas(y_true, pys_human,  "Pysentimiento  →  GT humano (B/N/M)")
m_xlm = metricas(y_true, xlm_human,  "XLM-RoBERTa    →  GT humano (B/N/M)")

# Valores do BERTimbau (já calculados nas células anteriores do notebook)
m_bert = {"Accuracy": 0.6667, "F1-macro": 0.5976, "F1-ponderado": 0.6707}

# ── 6. Tabela comparativa ─────────────────────────────────────────────────────
df_comp = pd.DataFrame({
    "Pysentimiento": m_pys,
    "XLM-RoBERTa":   m_xlm,
    "BERTimbau\n(nosso)": m_bert,
}).T.round(4)

print("\n" + "═"*60)
print("  TABELA COMPARATIVA — GT humano (conjunto de teste, n=300)")
print("═"*60)
print(df_comp.to_string())
print()

# ── 7. Matrizes de confusão ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(
    "Matrizes de confusão — GT humano (B/N/M), conjunto de teste (n=300)",
    fontsize=13, fontweight="bold"
)

configs = [
    (pys_human, "Pysentimiento"),
    (xlm_human, "XLM-RoBERTa"),
]

# BERTimbau — precisamos das predições; tenta reaproveitar variável do notebook
try:
    bert_pred_labels = [["B","N","M"][i] for i in y_pred_final]
    configs.append((bert_pred_labels, "BERTimbau (nosso)"))
except NameError:
    axes[2].set_title("BERTimbau\n(rode a célula 2.5 primeiro)")
    axes[2].axis("off")
    configs.append((None, None))

for ax, (preds, titulo) in zip(axes, configs):
    if preds is None:
        continue
    cm = confusion_matrix(y_true, preds, labels=LABELS_H)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABELS_H, yticklabels=LABELS_H,
        ax=ax, cbar=False
    )
    ax.set_title(titulo, fontweight="bold")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real (humano)")

plt.tight_layout()
plt.show()

# ── 8. Salva resultados num DataFrame para uso posterior ─────────────────────
df_teste["Pysentimiento_pred"] = pys_human
df_teste["XLM_pred"]           = xlm_human

print("\nPredições adicionadas ao DataFrame 'df_teste'.")
print("Colunas disponíveis:", df_teste.columns.tolist())
