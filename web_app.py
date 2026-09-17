# ============================================================
# NF1 — API DE PLN + IA
#
# ASSISTENTE VIRTUAL ACADÊMICO
#
# Tecnologias:
#
# Flask
# NLTK
# nltk.grammar
# scikit-learn
# pandas
#
# Técnicas:
#
# 1. Limpeza de texto
# 2. Normalização
# 3. Tokenização
# 4. Stopwords
# 5. Corpus textual
# 6. Bag of Words
# 7. Multinomial Naive Bayes
# 8. Probabilidades por classe
# 9. K-Means
# 10. Gramática formal
#
# A aplicação NÃO possui frontend.
#
# Comunicação:
#
# HTTP + JSON
#
# Exemplos:
#
# curl http://localhost:5000/
#
# curl http://localhost:5000/api/health
#
# curl http://localhost:5000/api/corpus
#
# curl -X POST http://localhost:5000/api/analisar \
#      -H "Content-Type: application/json" \
#      -d '{"texto":"Quando será a próxima prova?"}'
#
# ============================================================


# ============================================================
# IMPORTAÇÕES
# ============================================================

from flask import Flask, request, jsonify

from collections import Counter, defaultdict
import random

from rapidfuzz import fuzz, process
from nltk.util import ngrams

import re
import unicodedata

from collections import Counter

import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.grammar import CFG
from nltk.parse import ChartParser

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.cluster import KMeans


import re
import unicodedata


def normalizar_texto(texto):
    """
    Normaliza um texto para processamento de linguagem natural.

    Etapas:
    1. Converte para letras minúsculas.
    2. Remove acentos.
    3. Remove pontuação.
    4. Remove caracteres especiais.
    5. Remove espaços extras.
    """

    # Garantir que temos uma string
    texto = str(texto)

    # 1. Letras minúsculas
    texto = texto.lower()

    # 2. Remover acentos
    texto = unicodedata.normalize("NFD", texto)

    texto = "".join(
        caractere
        for caractere in texto
        if unicodedata.category(caractere) != "Mn"
    )

    # 3. Remover pontuação e caracteres especiais
    texto = re.sub(
        r"[^a-z0-9\s]",
        " ",
        texto
    )

    # 4. Remover espaços duplicados
    texto = re.sub(
        r"\s+",
        " ",
        texto
    )

    # 5. Remover espaços no início/fim
    texto = texto.strip()

    return texto

# ============================================================
# CONFIGURAÇÃO
# ============================================================

app = Flask(__name__)


# ============================================================
# NLTK
# ============================================================

def preparar_nltk():

    recursos = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords")
    ]

    for caminho, pacote in recursos:

        try:

            nltk.data.find(caminho)

        except LookupError:

            print(
                f"[NLTK] Baixando recurso: {pacote}"
            )

            nltk.download(
                pacote,
                quiet=True
            )


preparar_nltk()


STOPWORDS = set(
    stopwords.words("portuguese")
)


# ============================================================
# CORPUS
#
# Cada classe representa uma intenção.
#
# Esse corpus é pequeno propositalmente:
# objetivo didático.
# ============================================================

CORPUS = {

    "avaliacao": [

        "quando será a próxima prova",
        "qual a data da avaliação",
        "quando vai ser a prova",
        "qual o dia da prova",
        "quando será minha avaliação",
        "onde será a próxima prova",
        "quando começa a avaliação",
        "qual o prazo da prova",
        "quando teremos a avaliação",
        "qual a data da próxima prova",
        "quando vai acontecer a prova",
        "qual a data do exame",
        "quando será o exame",
        "quando teremos a prova",
        "qual o dia da avaliação"

    ],

    "notas": [

        "onde vejo minhas notas",
        "quero consultar minha nota",
        "qual foi minha nota",
        "quando sai a nota",
        "onde estão minhas avaliações",
        "quero saber minha nota",
        "como consultar minhas notas",
        "onde vejo o resultado da prova",
        "minhas notas já foram lançadas",
        "quero consultar minhas avaliações",
        "onde está minha nota",
        "quando será divulgada minha nota",
        "como vejo o resultado",
        "quero saber o resultado da avaliação",
        "onde consultar as notas"

    ],

    "frequencia": [

        "qual minha frequência",
        "quantas faltas eu tenho",
        "quero consultar minha frequência",
        "onde vejo minhas faltas",
        "qual minha porcentagem de presença",
        "quantas faltas posso ter",
        "como consultar minha presença",
        "quero saber minha frequência",
        "quantas aulas eu faltei",
        "minha frequência está disponível",
        "qual meu percentual de presença",
        "quero consultar minhas faltas",
        "quantas faltas tenho",
        "onde vejo minha frequência",
        "qual minha presença"

    ],

    "acesso": [

        "não consigo acessar o sistema",
        "minha senha não funciona",
        "não consigo entrar no brightspace",
        "problema para acessar a plataforma",
        "não consigo fazer login",
        "o sistema não abre",
        "não consigo acessar minha conta",
        "minha senha está errada",
        "não consigo entrar na plataforma",
        "problema no acesso ao sistema",
        "não consigo acessar o ambiente virtual",
        "não consigo entrar no sistema",
        "meu login não funciona",
        "problema com minha senha",
        "não consigo acessar a plataforma acadêmica"

    ],

        "financeiro": [

        "não consigo pagar minha mensalidade",
        "nao consigo gerar um boleto",
        "o valor da mensalidade esta errado",
        "quando vence a mensalidade?",
        "como vejo minhas faturas?"

    ],

    "calendario": [

        "quando começa o semestre",
        "qual o calendário acadêmico",
        "quando começam as aulas",
        "quando termina o semestre",
        "qual a data do próximo semestre",
        "quando será o início das aulas",
        "quais são as datas acadêmicas",
        "onde vejo o calendário",
        "qual o período letivo",
        "quando começam as atividades",
        "qual o calendário do semestre",
        "quando termina o período letivo",
        "quando começa o próximo semestre",
        "quais as datas das aulas",
        "onde encontro o calendário acadêmico"

    ]

}


# ============================================================
# RESPOSTAS
# ============================================================

RESPOSTAS = {

    "avaliacao":
        "Consulte o calendário acadêmico para verificar "
        "a data da próxima avaliação.",

    "notas":
        "As notas podem ser consultadas no ambiente "
        "acadêmico da instituição.",

    "frequencia":
        "A frequência pode ser consultada no sistema "
        "acadêmico na área do aluno.",

    "acesso":
        "Verifique seu usuário e senha. Caso o problema "
        "continue, entre em contato com o suporte acadêmico.",

    "calendario":
        "O calendário acadêmico contém as principais "
        "datas do período letivo.",

    "financeiro":
        "As faturas vencem dia 28 do mes"
        "bom dia, para esse assunto entre em contato com a tesouraria." 

}


# ============================================================
# GRAMÁTICA
#
# Gramática simplificada para fins didáticos.
#
# IMPORTANTE:
#
# Isto NÃO representa a gramática completa do português.
#
# Serve para demonstrar:
#
# linguagem natural
#        ↓
# tokens
#        ↓
# estrutura sintática
# ============================================================

GRAMATICA = CFG.fromstring("""

S -> PERGUNTA

PERGUNTA -> INTERROGATIVO VERBO COMPLEMENTO
PERGUNTA -> INTERROGATIVO COMPLEMENTO
PERGUNTA -> VERBO COMPLEMENTO

INTERROGATIVO -> 'quando'
INTERROGATIVO -> 'onde'
INTERROGATIVO -> 'qual'
INTERROGATIVO -> 'como'
INTERROGATIVO -> 'quantas'
INTERROGATIVO -> 'quais'

VERBO -> 'sera'
VERBO -> 'vai'
VERBO -> 'ver'
VERBO -> 'consultar'
VERBO -> 'comeca'
VERBO -> 'termina'

COMPLEMENTO -> 'prova'
COMPLEMENTO -> 'avaliacao'
COMPLEMENTO -> 'nota'
COMPLEMENTO -> 'notas'
COMPLEMENTO -> 'frequencia'
COMPLEMENTO -> 'faltas'
COMPLEMENTO -> 'calendario'
COMPLEMENTO -> 'semestre'

""")


PARSER = ChartParser(
    GRAMATICA
)


# ============================================================
# LIMPEZA
# ============================================================

def remover_acentos(texto):

    texto = unicodedata.normalize(
        "NFD",
        texto
    )

    texto = "".join(
        caractere
        for caractere in texto
        if unicodedata.category(caractere) != "Mn"
    )

    return texto


def limpar_texto(texto):

    texto = texto.lower()

    texto = remover_acentos(
        texto
    )

    texto = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        texto
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    )

    return texto.strip()


# ============================================================
# TOKENIZAÇÃO
# ============================================================

def tokenizar(texto):

    texto_limpo = limpar_texto(
        texto
    )

    tokens = word_tokenize(
        texto_limpo,
        language="portuguese"
    )

    return tokens


# ============================================================
# REMOÇÃO DE STOPWORDS
# ============================================================

def remover_stopwords(tokens):

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


# ============================================================
# DATASET
# ============================================================

textos = []

classes = []


for classe, exemplos in CORPUS.items():

    for exemplo in exemplos:

        textos.append(
            limpar_texto(exemplo)
        )

        classes.append(
            classe
        )


# ============================================================
# VETORIZAÇÃO
#
# Texto → números
#
# CountVectorizer produz Bag of Words.
#
# Exemplo:
#
# "prova" "nota" "frequencia"
#
#      ↓
#
# [1, 0, 0]
#
# ============================================================

vectorizer = CountVectorizer(
    lowercase=True
)


X = vectorizer.fit_transform(
    textos
)


# ============================================================
# MULTINOMIAL NAIVE BAYES
#
# MODELO SUPERVISIONADO
#
# Entrada:
# vetor de palavras
#
# Saída:
# intenção
# ============================================================

modelo_nb = MultinomialNB()

modelo_nb.fit(
    X,
    classes
)


# ============================================================
# K-MEANS
#
# MODELO NÃO SUPERVISIONADO
#
# O algoritmo não recebe as classes.
#
# Ele procura grupos de textos semelhantes.
# ============================================================

NUM_CLUSTERS = len(CORPUS)


modelo_kmeans = KMeans(
    n_clusters=NUM_CLUSTERS,
    random_state=42,
    n_init=10
)


clusters = modelo_kmeans.fit_predict(
    X
)


# ============================================================
# INTERPRETAÇÃO DIDÁTICA DOS CLUSTERS
#
# O K-Means não sabe que um cluster é "notas"
# ou "avaliação".
#
# Aqui calculamos a classe predominante apenas
# para facilitar a interpretação durante a aula.
# ============================================================

cluster_classes = {}


for cluster_id in range(NUM_CLUSTERS):

    indices = [
        i
        for i, cluster in enumerate(clusters)
        if cluster == cluster_id
    ]

    contagem = Counter(
        classes[i]
        for i in indices
    )

    if contagem:

        cluster_classes[cluster_id] = (
            contagem.most_common(1)[0][0]
        )

    else:

        cluster_classes[cluster_id] = None


# ============================================================
# FUNÇÃO:
# CLASSIFICAÇÃO NAIVE BAYES
# ============================================================

def classificar_naive_bayes(texto):

    texto_limpo = limpar_texto(
        texto
    )

    vetor = vectorizer.transform(
        [texto_limpo]
    )

    classe = modelo_nb.predict(
        vetor
    )[0]

    probabilidades = modelo_nb.predict_proba(
        vetor
    )[0]

    classes_modelo = modelo_nb.classes_

    probabilidades_dict = {}

    for nome_classe, probabilidade in zip(
        classes_modelo,
        probabilidades
    ):

        probabilidades_dict[nome_classe] = round(
            float(probabilidade),
            6
        )

    confianca = max(
        probabilidades_dict.values()
    )

    return {

        "classe": classe,

        "confianca": round(
            confianca,
            6
        ),

        "probabilidades": probabilidades_dict

    }


# ============================================================
# FUNÇÃO:
# K-MEANS
# ============================================================

def classificar_kmeans(texto):

    texto_limpo = limpar_texto(
        texto
    )

    vetor = vectorizer.transform(
        [texto_limpo]
    )

    cluster = int(
        modelo_kmeans.predict(
            vetor
        )[0]
    )

    classe_predominante = (
        cluster_classes.get(
            cluster
        )
    )

    distancias = (
        modelo_kmeans.transform(
            vetor
        )[0]
    )

    distancias_dict = {}

    for i, distancia in enumerate(
        distancias
    ):

        distancias_dict[str(i)] = round(
            float(distancia),
            6
        )

    return {

        "cluster": cluster,

        "classe_predominante_do_cluster":
            classe_predominante,

        "distancias": distancias_dict

    }


# ============================================================
# FUNÇÃO:
# ANÁLISE DA GRAMÁTICA
# ============================================================

def analisar_gramatica(tokens):

    # --------------------------------------------------------
    # A gramática foi construída com palavras sem acentos.
    # --------------------------------------------------------

    tokens_gramatica = [
        remover_acentos(token.lower())
        for token in tokens
    ]

    # --------------------------------------------------------
    # O parser exige que os tokens estejam no vocabulário
    # da gramática.
    # --------------------------------------------------------

    tokens_validos = [

        token

        for token in tokens_gramatica

        if token in {
            "quando",
            "onde",
            "qual",
            "como",
            "quantas",
            "quais",
            "sera",
            "vai",
            "ver",
            "consultar",
            "comeca",
            "termina",
            "prova",
            "avaliacao",
            "nota",
            "notas",
            "frequencia",
            "faltas",
            "calendario",
            "semestre"
        }

    ]

    if not tokens_validos:

        return {

            "reconhecida": False,

            "tokens_utilizados": [],

            "arvores": []

        }


    try:

        arvores = list(
            PARSER.parse(
                tokens_validos
            )
        )

    except ValueError:

        arvores = []


    arvores_texto = []

    for arvore in arvores[:3]:

        arvores_texto.append(
            arvore.pformat(
                margin=100
            )
        )


    return {

        "reconhecida":
            len(arvores) > 0,

        "tokens_utilizados":
            tokens_validos,

        "quantidade_arvores":
            len(arvores),

        "arvores":
            arvores_texto

    }


# ============================================================
# ANÁLISE COMPLETA
# ============================================================

def analisar_texto(texto):

    # --------------------------------------------------------
    # 1. LIMPEZA
    # --------------------------------------------------------

    texto_limpo = limpar_texto(
        texto
    )

    # --------------------------------------------------------
    # 2. TOKENIZAÇÃO
    # --------------------------------------------------------

    tokens = tokenizar(
        texto
    )

    # --------------------------------------------------------
    # 3. STOPWORDS
    # --------------------------------------------------------

    tokens_sem_stopwords = (
        remover_stopwords(
            tokens
        )
    )

    # --------------------------------------------------------
    # 4. NAIVE BAYES
    # --------------------------------------------------------

    resultado_nb = (
        classificar_naive_bayes(
            texto
        )
    )

    # --------------------------------------------------------
    # 5. K-MEANS
    # --------------------------------------------------------

    resultado_kmeans = (
        classificar_kmeans(
            texto
        )
    )

    # --------------------------------------------------------
    # 6. GRAMÁTICA
    # --------------------------------------------------------

    resultado_gramatica = (
        analisar_gramatica(
            tokens
        )
    )

    # --------------------------------------------------------
    # 7. RESPOSTA
    # --------------------------------------------------------

    intencao = (
        resultado_nb["classe"]
    )

    resposta = RESPOSTAS.get(
        intencao,
        "Não foi possível identificar uma resposta."
    )

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    return {

        "entrada": {

            "texto_original":
                texto,

            "texto_limpo":
                texto_limpo,

            "tokens":
                tokens,

            "tokens_sem_stopwords":
                tokens_sem_stopwords

        },

        "naive_bayes":
            resultado_nb,

        "kmeans":
            resultado_kmeans,

        "gramatica":
            resultado_gramatica,

        "resposta":
            resposta

    }


# ============================================================
# VALIDAÇÃO DA ENTRADA
# ============================================================

def obter_texto_json():

    if not request.is_json:

        return None, (

            jsonify({

                "erro":
                    "A requisição deve utilizar Content-Type: application/json"

            }),

            415

        )


    dados = request.get_json(
        silent=True
    )


    if not dados:

        return None, (

            jsonify({

                "erro":
                    "JSON inválido ou vazio."

            }),

            400

        )


    texto = dados.get(
        "texto"
    )


    if not isinstance(
        texto,
        str
    ):

        return None, (

            jsonify({

                "erro":
                    "O campo 'texto' deve ser uma string."

            }),

            400

        )


    texto = texto.strip()


    if not texto:

        return None, (

            jsonify({

                "erro":
                    "O campo 'texto' não pode estar vazio."

            }),

            400

        )


    return texto, None


# ============================================================
# ENDPOINT
# /
# ============================================================

@app.route("/api/nltk/analisar", methods=["POST"])
def nltk_analisar():

    dados = request.get_json(silent=True) or {}
    texto = dados.get("texto", "").strip()

    if not texto:
        return jsonify({
            "erro": "Informe o campo 'texto'."
        }), 400

    normalizado = normalizar_texto(texto)

    tokens = word_tokenize(normalizado, language="portuguese")

    tokens_sem_stopwords = [
        token
        for token in tokens
        if token not in STOPWORDS_PT
    ]

    return jsonify({
        "texto_original": texto,
        "normalizado": normalizado,
        "tokens": tokens,
        "sem_stopwords": tokens_sem_stopwords,
        "quantidade_tokens": len(tokens),
        "quantidade_sem_stopwords": len(tokens_sem_stopwords)
    })
    
@app.route("/api/nltk/frequencia", methods=["POST"])
def nltk_frequencia():

    dados = request.get_json(silent=True) or {}
    texto = dados.get("texto", "").strip()

    if not texto:
        return jsonify({
            "erro": "Informe o campo 'texto'."
        }), 400

    normalizado = normalizar_texto(texto)

    tokens = word_tokenize(
        normalizado,
        language="portuguese"
    )

    tokens = [
        token
        for token in tokens
        if token not in STOPWORDS_PT
    ]

    frequencias = Counter(tokens)

    return jsonify({
        "texto": texto,
        "frequencia": dict(
            sorted(
                frequencias.items(),
                key=lambda x: (-x[1], x[0])
            )
        )
    })    

@app.route("/api/nltk/ngramas", methods=["POST"])
def nltk_ngramas():

    dados = request.get_json(silent=True) or {}

    texto = dados.get("texto", "").strip()
    n = dados.get("n", 2)

    if not texto:
        return jsonify({
            "erro": "Informe o campo 'texto'."
        }), 400

    try:
        n = int(n)
    except (TypeError, ValueError):
        return jsonify({
            "erro": "O campo 'n' deve ser inteiro."
        }), 400

    if n < 1:
        return jsonify({
            "erro": "n deve ser maior ou igual a 1."
        }), 400

    normalizado = normalizar_texto(texto)

    tokens = word_tokenize(
        normalizado,
        language="portuguese"
    )

    resultado = list(ngrams(tokens, n))

    return jsonify({
        "texto": texto,
        "n": n,
        "quantidade": len(resultado),
        "ngramas": [
            list(item)
            for item in resultado
        ]
    })
    
@app.route("/api/fuzzy/similaridade", methods=["POST"])
def fuzzy_similaridade():

    dados = request.get_json(silent=True) or {}

    texto1 = dados.get("texto1", "").strip()
    texto2 = dados.get("texto2", "").strip()

    if not texto1 or not texto2:
        return jsonify({
            "erro": "Informe 'texto1' e 'texto2'."
        }), 400

    similaridade = fuzz.ratio(
        texto1.lower(),
        texto2.lower()
    )

    similaridade_parcial = fuzz.partial_ratio(
        texto1.lower(),
        texto2.lower()
    )

    similaridade_tokens = fuzz.token_sort_ratio(
        texto1.lower(),
        texto2.lower()
    )

    return jsonify({
        "texto1": texto1,
        "texto2": texto2,
        "similaridade": round(
            similaridade / 100,
            4
        ),
        "similaridade_parcial": round(
            similaridade_parcial / 100,
            4
        ),
        "similaridade_tokens": round(
            similaridade_tokens / 100,
            4
        )
    })    


@app.route("/api/fuzzy/comparar", methods=["POST"])
def fuzzy_comparar():

    dados = request.get_json(silent=True) or {}
    texto = dados.get("texto", "").strip()

    if not texto:
        return jsonify({
            "erro": "Informe o campo 'texto'."
        }), 400

    resultados = []

    for exemplo in CORPUS:

        frase = exemplo["texto"]
        intencao = exemplo["intencao"]

        similaridade = fuzz.token_sort_ratio(
            texto.lower(),
            frase.lower()
        )

        resultados.append({
            "texto": frase,
            "intencao": intencao,
            "similaridade": round(
                similaridade / 100,
                4
            )
        })

    resultados.sort(
        key=lambda x: x["similaridade"],
        reverse=True
    )

    return jsonify({
        "entrada": texto,
        "resultados": resultados[:5]
    })

@app.route("/api/fuzzy/intencao", methods=["POST"])
def fuzzy_intencao():

    dados = request.get_json(silent=True) or {}
    texto = dados.get("texto", "").strip()

    if not texto:
        return jsonify({
            "erro": "Informe o campo 'texto'."
        }), 400

    melhores = []

    for exemplo in CORPUS:

        similaridade = fuzz.token_sort_ratio(
            texto.lower(),
            exemplo["texto"].lower()
        )

        melhores.append({
            "texto": exemplo["texto"],
            "intencao": exemplo["intencao"],
            "similaridade": similaridade
        })

    melhores.sort(
        key=lambda x: x["similaridade"],
        reverse=True
    )

    melhor = melhores[0]

    return jsonify({
        "entrada": texto,
        "intencao": melhor["intencao"],
        "frase_mais_proxima": melhor["texto"],
        "similaridade": round(
            melhor["similaridade"] / 100,
            4
        ),
        "top_5": [
            {
                **item,
                "similaridade": round(
                    item["similaridade"] / 100,
                    4
                )
            }
            for item in melhores[:5]
        ]
    })

@app.route("/api/probabilidades", methods=["POST"])
def probabilidades():

    dados = request.get_json(silent=True) or {}
    texto = dados.get("texto", "").strip()

    if not texto:
        return jsonify({
            "erro": "Informe o campo 'texto'."
        }), 400

    texto_normalizado = normalizar_texto(texto)

    X = vectorizer.transform([
        texto_normalizado
    ])

    probabilidades = modelo_nb.predict_proba(X)[0]

    resultado = {}

    for classe, probabilidade in zip(
        modelo_nb.classes_,
        probabilidades
    ):
        resultado[classe] = round(
            float(probabilidade),
            6
        )

    return jsonify({
        "texto": texto,
        "probabilidades": resultado,
        "classe_predita": modelo_nb.predict(X)[0]
    })


def construir_cadeia_markov():

    transicoes = defaultdict(Counter)

    for exemplo in CORPUS:

        texto = normalizar_texto(
            exemplo["texto"]
        )

        tokens = word_tokenize(
            texto,
            language="portuguese"
        )

        for atual, proxima in zip(
            tokens,
            tokens[1:]
        ):
            transicoes[atual][proxima] += 1

    return transicoes    

@app.route("/api/markov/transicoes", methods=["GET"])
def markov_transicoes():

    resultado = {}

    for palavra, destinos in MARKOV.items():

        total = sum(destinos.values())

        resultado[palavra] = {
            destino: round(
                quantidade / total,
                4
            )
            for destino, quantidade
            in destinos.items()
        }

    return jsonify({
        "transicoes": resultado
    })
    
@app.route("/api/markov/proximo", methods=["POST"])
def markov_proximo():

    dados = request.get_json(silent=True) or {}

    palavra = normalizar_texto(
        dados.get("palavra", "").strip()
    )

    if not palavra:
        return jsonify({
            "erro": "Informe a palavra."
        }), 400

    if palavra not in MARKOV:
        return jsonify({
            "palavra_atual": palavra,
            "proximas_palavras": {},
            "mensagem": "Palavra não encontrada na cadeia."
        })

    destinos = MARKOV[palavra]

    total = sum(destinos.values())

    probabilidades = {
        destino: round(
            quantidade / total,
            4
        )
        for destino, quantidade
        in destinos.items()
    }

    return jsonify({
        "palavra_atual": palavra,
        "proximas_palavras": probabilidades
    })

@app.route("/api/markov/gerar", methods=["POST"])
def markov_gerar():

    dados = request.get_json(silent=True) or {}

    inicio = normalizar_texto(
        dados.get("inicio", "").strip()
    )

    tamanho = dados.get("tamanho", 10)

    if not inicio:
        return jsonify({
            "erro": "Informe 'inicio'."
        }), 400

    try:
        tamanho = int(tamanho)
    except (TypeError, ValueError):
        tamanho = 10

    if tamanho < 1:
        tamanho = 1

    palavras = [inicio]

    atual = inicio

    for _ in range(tamanho - 1):

        if atual not in MARKOV:
            break

        destinos = MARKOV[atual]

        palavras_possiveis = list(
            destinos.keys()
        )

        pesos = list(
            destinos.values()
        )

        proxima = random.choices(
            palavras_possiveis,
            weights=pesos,
            k=1
        )[0]

        palavras.append(proxima)

        atual = proxima

    return jsonify({
        "inicio": inicio,
        "tamanho_solicitado": tamanho,
        "texto_gerado": " ".join(palavras),
        "tokens": palavras
    })

@app.route("/api/modelos", methods=["GET"])
def modelos():

    return jsonify({
        "modelos": [
            {
                "nome": "NLTK",
                "tipo": "processamento linguistico",
                "funcao": "tokenizacao, stopwords, frequencia e n-gramas"
            },
            {
                "nome": "Fuzzy Matching",
                "tipo": "similaridade",
                "funcao": "comparacao de strings"
            },
            {
                "nome": "Multinomial Naive Bayes",
                "tipo": "aprendizado supervisionado",
                "funcao": "classificacao de texto"
            },
            {
                "nome": "Gramática NLTK",
                "tipo": "regras",
                "funcao": "analise sintatica"
            },
            {
                "nome": "Cadeia de Markov",
                "tipo": "modelo probabilistico",
                "funcao": "modelagem de transicoes entre palavras"
            },
            {
                "nome": "K-Means",
                "tipo": "aprendizado nao supervisionado",
                "funcao": "agrupamento"
            }
        ]
    })    

@app.route("/api/pipeline", methods=["POST"])
def pipeline():

    dados = request.get_json(silent=True) or {}

    texto = dados.get("texto", "").strip()

    if not texto:
        return jsonify({
            "erro": "Informe o campo 'texto'."
        }), 400

    # ========================================================
    # 1. NLTK
    # ========================================================

    normalizado = normalizar_texto(texto)

    tokens = word_tokenize(
        normalizado,
        language="portuguese"
    )

    tokens_sem_stopwords = [
        token
        for token in tokens
        if token not in STOPWORDS_PT
    ]

    # ========================================================
    # 2. NAIVE BAYES
    # ========================================================

    X = vectorizer.transform([
        normalizado
    ])

    classe = modelo_nb.predict(X)[0]

    probabilidades = modelo_nb.predict_proba(X)[0]

    probabilidades_dict = {
        classe_modelo: round(
            float(probabilidade),
            4
        )
        for classe_modelo, probabilidade
        in zip(
            modelo_nb.classes_,
            probabilidades
        )
    }

    # ========================================================
    # 3. FUZZY
    # ========================================================

    fuzzy_resultados = []

    for exemplo in CORPUS:

        similaridade = fuzz.token_sort_ratio(
            texto.lower(),
            exemplo["texto"].lower()
        )

        fuzzy_resultados.append({
            "texto": exemplo["texto"],
            "intencao": exemplo["intencao"],
            "similaridade": round(
                similaridade / 100,
                4
            )
        })

    fuzzy_resultados.sort(
        key=lambda x: x["similaridade"],
        reverse=True
    )

    melhor_fuzzy = fuzzy_resultados[0]

    # ========================================================
    # 4. RESPOSTA
    # ========================================================

    resposta = RESPOSTAS.get(
        classe,
        "Não encontrei uma resposta para essa mensagem."
    )

    # ========================================================
    # RESULTADO
    # ========================================================

    return jsonify({

        "entrada": texto,

        "nltk": {
            "normalizado": normalizado,
            "tokens": tokens,
            "sem_stopwords": tokens_sem_stopwords
        },

        "naive_bayes": {
            "intencao": classe,
            "probabilidades": probabilidades_dict
        },

        "fuzzy": {
            "frase_mais_proxima":
                melhor_fuzzy["texto"],
            "intencao":
                melhor_fuzzy["intencao"],
            "similaridade":
                melhor_fuzzy["similaridade"]
        },

        "resposta": resposta
    })

@app.route(
    "/",
    methods=["GET"]
)
def inicio():

    return jsonify({

        "aplicacao":
            "NF1 — API de PLN + IA",

        "descricao":
            "Assistente Virtual Acadêmico",

        "modelo_supervisionado":
            "Multinomial Naive Bayes",

        "modelo_nao_supervisionado":
            "K-Means",

        "processamento":
            [

                "limpeza",

                "normalização",

                "tokenização",

                "stopwords",

                "Bag of Words",

                "classificação",

                "agrupamento",

                "gramática formal"

            ],

        "endpoints":
            [

                "GET /",

                "GET /api/health",

                "GET /api/info",

                "GET /api/corpus",

                "GET /api/vocabulario",

                "GET /api/clusters",

                "POST /api/analisar",

                "POST /api/classificar",

                "POST /api/tokenizar",

                "POST /api/gramatica"

            ]

    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "ok",

        "modelo":
            "Multinomial Naive Bayes",

        "corpus":
            "carregado",

        "quantidade_documentos":
            len(textos),

        "quantidade_classes":
            len(CORPUS)

    })


# ============================================================
# INFORMAÇÕES DO MODELO
# ============================================================

@app.route(
    "/api/info",
    methods=["GET"]
)
def info():

    return jsonify({

        "modelo": {

            "tipo":
                "classificação supervisionada",

            "algoritmo":
                "Multinomial Naive Bayes",

            "representacao":
                "Bag of Words",

            "quantidade_documentos":
                len(textos),

            "quantidade_caracteristicas":
                len(
                    vectorizer.get_feature_names_out()
                ),

            "classes":
                list(
                    modelo_nb.classes_
                )

        },

        "kmeans": {

            "tipo":
                "aprendizado não supervisionado",

            "algoritmo":
                "K-Means",

            "clusters":
                NUM_CLUSTERS

        },

        "nltk": {

            "tokenizacao":
                "word_tokenize",

            "stopwords":
                "portuguese",

            "gramatica":
                "CFG",

            "parser":
                "ChartParser"

        }

    })


# ============================================================
# CORPUS
# ============================================================

@app.route(
    "/api/corpus",
    methods=["GET"]
)
def corpus_api():

    quantidade_por_classe = {

        classe:
            len(exemplos)

        for classe, exemplos
        in CORPUS.items()

    }

    return jsonify({

        "quantidade_total":
            len(textos),

        "quantidade_por_classe":
            quantidade_por_classe,

        "classes":
            list(CORPUS.keys()),

        "corpus":
            CORPUS

    })


# ============================================================
# VOCABULÁRIO
# ============================================================

@app.route(
    "/api/vocabulario",
    methods=["GET"]
)
def vocabulario():

    palavras = (
        vectorizer
        .get_feature_names_out()
        .tolist()
    )

    return jsonify({

        "quantidade":
            len(palavras),

        "palavras":
            palavras

    })


# ============================================================
# CLUSTERS
# ============================================================

@app.route(
    "/api/clusters",
    methods=["GET"]
)
def clusters_api():

    resultado = []


    for i, texto in enumerate(
        textos
    ):

        cluster = int(
            clusters[i]
        )

        resultado.append({

            "documento":
                texto,

            "classe_real":
                classes[i],

            "cluster":
                cluster,

            "classe_predominante_cluster":
                cluster_classes.get(
                    cluster
                )

        })


    return jsonify({

        "quantidade_clusters":
            NUM_CLUSTERS,

        "clusters":
            resultado

    })


# ============================================================
# TOKENIZAÇÃO
# ============================================================

@app.route(
    "/api/tokenizar",
    methods=["POST"]
)
def tokenizar_api():

    texto, erro = (
        obter_texto_json()
    )


    if erro:

        return erro


    tokens = tokenizar(
        texto
    )

    tokens_sem_stopwords = (
        remover_stopwords(
            tokens
        )
    )


    return jsonify({

        "texto_original":
            texto,

        "texto_limpo":
            limpar_texto(
                texto
            ),

        "tokens":
            tokens,

        "tokens_sem_stopwords":
            tokens_sem_stopwords,

        "quantidade_tokens":
            len(tokens),

        "quantidade_tokens_sem_stopwords":
            len(tokens_sem_stopwords)

    })


# ============================================================
# CLASSIFICAÇÃO
#
# Apenas Naive Bayes + K-Means.
# ============================================================

@app.route(
    "/api/classificar",
    methods=["POST"]
)
def classificar_api():

    texto, erro = (
        obter_texto_json()
    )


    if erro:

        return erro


    resultado_nb = (
        classificar_naive_bayes(
            texto
        )
    )


    resultado_kmeans = (
        classificar_kmeans(
            texto
        )
    )


    return jsonify({

        "texto":
            texto,

        "naive_bayes":
            resultado_nb,

        "kmeans":
            resultado_kmeans

    })


# ============================================================
# GRAMÁTICA
# ============================================================

@app.route(
    "/api/gramatica",
    methods=["POST"]
)
def gramatica_api():

    texto, erro = (
        obter_texto_json()
    )


    if erro:

        return erro


    tokens = tokenizar(
        texto
    )


    resultado = (
        analisar_gramatica(
            tokens
        )
    )


    return jsonify({

        "texto":
            texto,

        "tokens":
            tokens,

        "gramatica":
            resultado

    })


# ============================================================
# ANÁLISE COMPLETA
#
# ENDPOINT PRINCIPAL
# ============================================================

@app.route(
    "/api/analisar",
    methods=["POST"]
)
def analisar_api():

    texto, erro = (
        obter_texto_json()
    )


    if erro:

        return erro


    resultado = (
        analisar_texto(
            texto
        )
    )


    return jsonify(
        resultado
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("NF1 — API DE PLN + IA")
    print("=" * 60)

    print()
    print("Modelo:")
    print("Multinomial Naive Bayes")

    print()
    print("Agrupamento:")
    print("K-Means")

    print()
    print("Gramática:")
    print("NLTK CFG + ChartParser")

    print()
    print("Documentos:")
    print(len(textos))

    print()
    print("Vocabulário:")
    print(
        len(
            vectorizer
            .get_feature_names_out()
        )
    )

    print()
    print("Classes:")

    for classe in CORPUS:

        print(
            f"  - {classe}"
        )

    print()
    print("Servidor:")
    print("http://127.0.0.1:5000")

    print()
    print("Exemplo:")

    print(
        'curl -X POST http://127.0.0.1:5000/api/analisar '
        '-H "Content-Type: application/json" '
        '-d \'{"texto":"Quando será a próxima prova?"}\''
    )

    print()
    print("=" * 60)
    print()


    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
