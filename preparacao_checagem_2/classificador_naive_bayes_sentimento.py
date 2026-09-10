# ============================================================
# CLASSIFICADOR DE SENTIMENTOS
# Naive Bayes + Bag of Words
# Corpus Movie Reviews - NLTK
# ============================================================

import nltk
import random
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from nltk.corpus import movie_reviews

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. DOWNLOAD DO CORPUS
# ============================================================

nltk.download("movie_reviews")


# ============================================================
# 2. CARREGAR O CORPUS
# ============================================================

arquivos = movie_reviews.fileids()

print("=" * 60)
print("1. CORPUS")
print("=" * 60)

print("Total de documentos:", len(arquivos))

print("Avaliações positivas:",
      len(movie_reviews.fileids("pos")))

print("Avaliações negativas:",
      len(movie_reviews.fileids("neg")))


# ============================================================
# 3. PREPARAR OS TEXTOS
# ============================================================

def preparar_texto(arquivo):

    palavras = movie_reviews.words(arquivo)

    # Minúsculas + somente palavras alfabéticas
    palavras = [
        palavra.lower()
        for palavra in palavras
        if palavra.isalpha()
    ]

    return " ".join(palavras)


textos = []
rotulos = []


for arquivo in arquivos:

    textos.append(
        preparar_texto(arquivo)
    )

    if arquivo.startswith("pos"):
        rotulos.append("positivo")
    else:
        rotulos.append("negativo")


print("\nExemplo de texto:")
print(textos[0][:500])

print("\nClasse:")
print(rotulos[0])


# ============================================================
# 4. SEPARAR TREINO E TESTE
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    textos,
    rotulos,
    test_size=0.20,
    random_state=42,
    stratify=rotulos
)


print("\n" + "=" * 60)
print("2. TREINAMENTO E TESTE")
print("=" * 60)

print("Documentos para treinamento:", len(X_train))
print("Documentos para teste:", len(X_test))


# ============================================================
# 5. BAG OF WORDS
# ============================================================

vectorizer = CountVectorizer(
    min_df=2
)

X_train_bow = vectorizer.fit_transform(X_train)

X_test_bow = vectorizer.transform(X_test)


print("\n" + "=" * 60)
print("3. BAG OF WORDS")
print("=" * 60)

print("Quantidade de palavras no vocabulário:",
      len(vectorizer.vocabulary_))

print("Formato da matriz de treinamento:",
      X_train_bow.shape)

print("Formato da matriz de teste:",
      X_test_bow.shape)


# ============================================================
# 6. VISUALIZAR PARTE DO VOCABULÁRIO
# ============================================================

print("\nPrimeiras palavras do vocabulário:")

palavras_vocabulario = list(
    vectorizer.vocabulary_.keys()
)

print(
    palavras_vocabulario[:30]
)


# ============================================================
# 7. TREINAR O NAIVE BAYES
# ============================================================

modelo = MultinomialNB()

modelo.fit(
    X_train_bow,
    y_train
)


print("\n" + "=" * 60)
print("4. MODELO")
print("=" * 60)

print("Modelo:")
print(modelo)


# ============================================================
# 8. FAZER PREDIÇÕES
# ============================================================

y_pred = modelo.predict(
    X_test_bow
)


print("\n" + "=" * 60)
print("5. PREDIÇÕES")
print("=" * 60)

for i in range(10):

    print(
        f"Real: {y_test[i]:<10} "
        f"Predito: {y_pred[i]}"
    )


# ============================================================
# 9. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("6. ACCURACY")
print("=" * 60)

print(
    f"Accuracy: {accuracy:.2%}"
)


# ============================================================
# 10. PRECISION, RECALL E F1
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    pos_label="positivo"
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="positivo"
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="positivo"
)


print("\n" + "=" * 60)
print("7. OUTRAS MÉTRICAS")
print("=" * 60)

print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-score:  {f1:.2%}")


# ============================================================
# 11. RELATÓRIO COMPLETO
# ============================================================

print("\n" + "=" * 60)
print("8. CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 12. MATRIZ DE CONFUSÃO
# ============================================================

matriz = confusion_matrix(
    y_test,
    y_pred,
    labels=["negativo", "positivo"]
)

print("\n" + "=" * 60)
print("9. MATRIZ DE CONFUSÃO")
print("=" * 60)

print(matriz)


# ============================================================
# 13. PLOT DA MATRIZ DE CONFUSÃO
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=["Negativo", "Positivo"]
)

disp.plot()

plt.title("Matriz de Confusão - Naive Bayes")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")

plt.show()


# ============================================================
# 14. GRÁFICO DAS MÉTRICAS
# ============================================================

metricas = [
    accuracy,
    precision,
    recall,
    f1
]

nomes = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score"
]

plt.figure(figsize=(8, 5))

plt.bar(
    nomes,
    metricas
)

plt.ylim(0, 1)

plt.ylabel("Valor")
plt.title("Métricas do Classificador")

for i, valor in enumerate(metricas):

    plt.text(
        i,
        valor + 0.02,
        f"{valor:.2%}",
        ha="center"
    )

plt.show()


# ============================================================
# 15. DISTRIBUIÇÃO DAS CLASSES
# ============================================================

contagem_classes = Counter(rotulos)

plt.figure(figsize=(6, 5))

plt.bar(
    ["Negativo", "Positivo"],
    [
        contagem_classes["negativo"],
        contagem_classes["positivo"]
    ]
)

plt.ylabel("Quantidade de documentos")
plt.title("Distribuição das Classes")

plt.show()


# ============================================================
# 16. PALAVRAS MAIS FREQUENTES DO CORPUS
# ============================================================

todas_palavras = []

for texto in textos:

    todas_palavras.extend(
        texto.split()
    )


frequencias = Counter(
    todas_palavras
)

mais_frequentes = frequencias.most_common(15)

palavras = [
    item[0]
    for item in mais_frequentes
]

quantidades = [
    item[1]
    for item in mais_frequentes
]


plt.figure(figsize=(10, 5))

plt.bar(
    palavras,
    quantidades
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.ylabel("Frequência")
plt.title("15 Palavras Mais Frequentes")

plt.tight_layout()

plt.show()


# ============================================================
# 17. TESTAR NOVAS FRASES
# ============================================================

novos_textos = [
    "this movie was excellent and amazing",
    "this movie was terrible and boring",
    "great film with excellent acting",
    "bad movie with terrible acting"
]


novos_bow = vectorizer.transform(
    novos_textos
)

novas_predicoes = modelo.predict(
    novos_bow
)


print("\n" + "=" * 60)
print("10. TESTANDO NOVOS TEXTOS")
print("=" * 60)

for texto, predicao in zip(
    novos_textos,
    novas_predicoes
):

    print("\nTexto:")
    print(texto)

    print("Sentimento:")
    print(predicao)


# ============================================================
# 18. PROBABILIDADE DAS CLASSES
# ============================================================

probabilidades = modelo.predict_proba(
    novos_bow
)

classes = modelo.classes_


print("\n" + "=" * 60)
print("11. PROBABILIDADES DO NAIVE BAYES")
print("=" * 60)

for texto, probs in zip(
    novos_textos,
    probabilidades
):

    print("\nTexto:")
    print(texto)

    for classe, probabilidade in zip(
        classes,
        probs
    ):

        print(
            f"P({classe}) = "
            f"{probabilidade:.4f}"
        )
