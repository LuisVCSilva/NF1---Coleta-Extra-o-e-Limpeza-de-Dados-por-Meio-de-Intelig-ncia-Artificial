import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def carregar_dados(caminho):
    with Path(caminho).open(encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    if "entradas" not in dados or not dados["entradas"]:
        raise ValueError("O arquivo JSON precisa conter uma lista 'entradas' com dados.")

    return dados["entradas"]


def obter_previsao(entrada, modelo):
    if modelo == "naive_bayes":
        return entrada[modelo]["classe"]
    return entrada[modelo]["classe_predominante_do_cluster"]


def construir_matriz(entradas, modelo):
    classes = sorted(
        {
            entrada["classe_real"]
            for entrada in entradas
        }
        | {
            obter_previsao(entrada, modelo)
            for entrada in entradas
        }
    )
    indices = {classe: indice for indice, classe in enumerate(classes)}
    matriz = [[0 for _ in classes] for _ in classes]

    for entrada in entradas:
        classe_real = entrada["classe_real"]
        classe_prevista = obter_previsao(entrada, modelo)
        matriz[indices[classe_real]][indices[classe_prevista]] += 1

    return classes, matriz


def plotar_matriz(classes, matriz, modelo, caminho_saida):
    figura, eixo = plt.subplots(figsize=(8, 6))
    imagem = eixo.imshow(matriz, cmap="Blues")
    figura.colorbar(imagem, ax=eixo, label="Quantidade")

    eixo.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Classe prevista",
        ylabel="Classe real",
        title=f"Matriz de confusão - {modelo}",
    )
    eixo.tick_params(axis="x", rotation=45)

    maior_valor = max(max(linha) for linha in matriz)
    for linha, valores in enumerate(matriz):
        for coluna, valor in enumerate(valores):
            cor = "white" if valor > maior_valor / 2 else "black"
            eixo.text(coluna, linha, valor, ha="center", va="center", color=cor)

    figura.tight_layout()
    figura.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    plt.close(figura)


def main():
    parser = argparse.ArgumentParser(
        description="Gera um PNG da matriz de confusão a partir de um arquivo JSON."
    )
    parser.add_argument("arquivo", help="Arquivo JSON com as entradas classificadas.")
    parser.add_argument(
        "-m",
        "--modelo",
        choices=("naive_bayes", "kmeans"),
        default="naive_bayes",
        help="Classificador usado na matriz (padrão: naive_bayes).",
    )
    parser.add_argument(
        "-o",
        "--saida",
        default="matriz_confusao.png",
        help="Caminho do PNG de saída (padrão: matriz_confusao.png).",
    )
    argumentos = parser.parse_args()

    entradas = carregar_dados(argumentos.arquivo)
    classes, matriz = construir_matriz(entradas, argumentos.modelo)
    plotar_matriz(classes, matriz, argumentos.modelo, argumentos.saida)
    print(f"Matriz de confusão salva em: {argumentos.saida}")


if __name__ == "__main__":
    main()