# ============================================================
# BAG OF WORDS (BoW) COM NLTK
# Corpus: Movie Reviews
# ============================================================

import nltk
from collections import Counter

# Execute uma vez no Google Colab:
nltk.download("movie_reviews")

from nltk.corpus import movie_reviews


# ============================================================
# 1. CORPUS
# ============================================================

print("=" * 60)
print("1. CORPUS")
print("=" * 60)

arquivos = movie_reviews.fileids()

print("Quantidade de documentos:", len(arquivos))

print("Primeiros documentos:")
print(arquivos[:5])


# ============================================================
# 2. ESCOLHER UM DOCUMENTO
# ============================================================

print("\n" + "=" * 60)
print("2. DOCUMENTO")
print("=" * 60)

arquivo = arquivos[0]

print("Arquivo:", arquivo)
print("Classe:", movie_reviews.categories(arquivo))

palavras_originais = movie_reviews.words(arquivo)

print("\nPrimeiras palavras:")
print(palavras_originais[:30])


# ============================================================
# 3. TOKENIZAÇÃO
# ============================================================

print("\n" + "=" * 60)
print("3. TOKENIZAÇÃO")
print("=" * 60)

# O corpus já fornece as palavras separadas em tokens.
tokens = list(palavras_originais)

print("Quantidade de tokens:", len(tokens))

print("Exemplo:")
print(tokens[:20])


# ============================================================
# 4. NORMALIZAÇÃO
# ============================================================

print("\n" + "=" * 60)
print("4. NORMALIZAÇÃO")
print("=" * 60)

# Transformar todas as palavras em minúsculas
tokens_lower = [
    token.lower()
    for token in tokens
]

print("Antes:")
print(tokens[:20])

print("\nDepois:")
print(tokens_lower[:20])


# ============================================================
# 5. REMOVER TOKENS NÃO ALFABÉTICOS
# ============================================================

print("\n" + "=" * 60)
print("5. FILTRO ALFABÉTICO")
print("=" * 60)

tokens_limpos = [
    token
    for token in tokens_lower
    if token.isalpha()
]

print("Tokens antes:", len(tokens_lower))
print("Tokens depois:", len(tokens_limpos))

print("\nExemplo de tokens limpos:")
print(tokens_limpos[:30])


# ============================================================
# 6. FREQUÊNCIA DAS PALAVRAS
# ============================================================

print("\n" + "=" * 60)
print("6. FREQUÊNCIA DAS PALAVRAS")
print("=" * 60)

frequencias = Counter(tokens_limpos)

print("Palavras mais frequentes:")

for palavra, quantidade in frequencias.most_common(10):
    print(f"{palavra:<15} {quantidade}")


# ============================================================
# 7. VOCABULÁRIO
# ============================================================

print("\n" + "=" * 60)
print("7. VOCABULÁRIO")
print("=" * 60)

# Conjunto das palavras diferentes
vocabulario = sorted(set(tokens_limpos))

print("Quantidade de palavras diferentes:")
print(len(vocabulario))

print("\nPrimeiras palavras do vocabulário:")
print(vocabulario[:30])


# ============================================================
# 8. BAG OF WORDS
# ============================================================

print("\n" + "=" * 60)
print("8. BAG OF WORDS")
print("=" * 60)

# Vamos escolher um pequeno vocabulário
vocabulario_bow = [
    "good",
    "bad",
    "movie",
    "film",
    "great"
]

print("Vocabulário utilizado:")
print(vocabulario_bow)


# ============================================================
# 9. PRESENÇA / AUSÊNCIA
# ============================================================

print("\n" + "=" * 60)
print("9. PRESENÇA / AUSÊNCIA")
print("=" * 60)

print("Palavra       Presente?")

for palavra in vocabulario_bow:

    presente = palavra in frequencias

    print(
        f"{palavra:<14} {presente}"
    )


# ============================================================
# 10. CONTAGEM
# ============================================================

print("\n" + "=" * 60)
print("10. CONTAGEM")
print("=" * 60)

print("Palavra       Quantidade")

for palavra in vocabulario_bow:

    quantidade = frequencias[palavra]

    print(
        f"{palavra:<14} {quantidade}"
    )


# ============================================================
# 11. VETOR BAG OF WORDS
# ============================================================

print("\n" + "=" * 60)
print("11. VETOR BAG OF WORDS")
print("=" * 60)

vetor_bow = [
    frequencias[palavra]
    for palavra in vocabulario_bow
]

print("Vocabulário:")
print(vocabulario_bow)

print("\nVetor:")
print(vetor_bow)


# ============================================================
# 12. MOSTRAR A RELAÇÃO PALAVRA -> POSIÇÃO -> VALOR
# ============================================================

print("\n" + "=" * 60)
print("12. INTERPRETAÇÃO DO VETOR")
print("=" * 60)

print("Posição   Palavra       Contagem")

for i, palavra in enumerate(vocabulario_bow):

    print(
        f"{i:<9} {palavra:<13} {frequencias[palavra]}"
    )


# ============================================================
# 13. COMPARAR DOIS DOCUMENTOS
# ============================================================

print("\n" + "=" * 60)
print("13. DOIS DOCUMENTOS")
print("=" * 60)

arquivo1 = movie_reviews.fileids("pos")[0]
arquivo2 = movie_reviews.fileids("neg")[0]

print("Documento 1:", arquivo1)
print("Classe:", movie_reviews.categories(arquivo1))

print("\nDocumento 2:", arquivo2)
print("Classe:", movie_reviews.categories(arquivo2))


# ============================================================
# 14. TRANSFORMAR OS DOIS DOCUMENTOS EM BoW
# ============================================================

print("\n" + "=" * 60)
print("14. BoW DOS DOIS DOCUMENTOS")
print("=" * 60)

def preparar_documento(arquivo):

    tokens = movie_reviews.words(arquivo)

    tokens = [
        token.lower()
        for token in tokens
        if token.isalpha()
    ]

    return Counter(tokens)


contagem1 = preparar_documento(arquivo1)
contagem2 = preparar_documento(arquivo2)


vetor1 = [
    contagem1[palavra]
    for palavra in vocabulario_bow
]

vetor2 = [
    contagem2[palavra]
    for palavra in vocabulario_bow
]


print("Vocabulário:")
print(vocabulario_bow)

print("\nDocumento 1:")
print(vetor1)

print("\nDocumento 2:")
print(vetor2)


# ============================================================
# 15. IDEIA CENTRAL DO BoW
# ============================================================

print("\n" + "=" * 60)
print("15. IDEIA CENTRAL")
print("=" * 60)
