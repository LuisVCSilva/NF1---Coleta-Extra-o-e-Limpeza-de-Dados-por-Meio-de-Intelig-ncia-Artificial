# Atividade Prática: Análise de Dados e Machine Learning com Imóveis de São Paulo

## Disciplina: Ciência de Dados e Inteligência Artificial

## Dataset

Nesta atividade será utilizado o dataset:

**São Paulo Real Estate Sales and Rentals 2020-2026**

O conjunto de dados contém informações sobre imóveis da cidade de São Paulo, incluindo características físicas, localização, valores de venda e aluguel, custos associados e indicadores econômicos.

O objetivo da atividade é desenvolver um fluxo completo de análise de dados utilizando técnicas de Ciência de Dados e Aprendizado de Máquina.

Durante a atividade serão trabalhadas as seguintes etapas:

- Coleta de dados;
- Exploração e análise inicial;
- Limpeza e preparação dos dados;
- Visualização dos dados;
- Clusterização utilizando K-Means;
- Construção de modelo preditivo utilizando Árvore de Decisão;
- Avaliação e interpretação dos resultados.


---

# 1. Preparação do ambiente

A atividade deverá ser realizada utilizando o Google Colab.

Inicialmente, instale a biblioteca responsável pelo download dos dados do Kaggle:

```python
!pip install kagglehub
```

Importe as bibliotecas necessárias:

```python
import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
```

---

# 2. Download do dataset

Utilize o código abaixo para realizar o download do conjunto de dados:

```python
path = kagglehub.dataset_download(
    "sergionefedov/so-paulo-real-estate-sales-and-rentals-2020-2026"
)

print("Path:", path)

print("\nArquivos:")
print(os.listdir(path))
```

Após a execução, observe:

- O caminho onde o dataset foi armazenado;
- Os arquivos disponíveis;
- O formato dos dados.


---

# 3. Carregamento dos dados

Realize a leitura do arquivo utilizando o Pandas.

Exemplo:

```python
arquivo = os.listdir(path)[0]

df = pd.read_csv(
    os.path.join(path, arquivo)
)

df.head()
```

Após carregar os dados, responda:

1. Quantas linhas e colunas existem no dataset?
2. Quais são os nomes das variáveis disponíveis?
3. Quais variáveis representam características do imóvel?
4. Qual variável representa o preço do imóvel?


---

# 4. Análise inicial dos dados

Execute uma análise exploratória inicial:

```python
df.info()

df.describe()
```

Investigue:

- Tipos de dados existentes;
- Variáveis numéricas e categóricas;
- Valores mínimos e máximos;
- Possíveis inconsistências.


Entregue:

- Uma descrição das principais características do dataset;
- Uma tabela contendo as principais estatísticas das variáveis numéricas.


---

# 5. Análise de valores ausentes

Verifique a existência de dados incompletos:

```python
df.isnull().sum().sort_values(
    ascending=False
)
```

Responda:

1. Quais variáveis possuem maior quantidade de valores ausentes?
2. Qual estratégia será utilizada para tratar esses valores?

Possíveis estratégias:

- Remover registros incompletos;
- Preencher valores utilizando média ou mediana;
- Utilizar técnicas específicas para variáveis categóricas.


---

# 6. Visualização dos dados

Construa gráficos para compreender o comportamento dos imóveis.


## 6.1 Distribuição dos preços

Crie um histograma utilizando a variável:

```
price_brl
```

Analise:

- O comportamento dos preços;
- Existência de valores extremos;
- Concentração dos imóveis.


---

## 6.2 Relação entre área e preço

Construa um gráfico de dispersão utilizando:

Eixo X:

```
area_util_m2
```

Eixo Y:

```
price_brl
```

Analise:

- Existe relação entre tamanho do imóvel e preço?
- Existem imóveis fora do padrão?


---

# 7. Preparação dos dados para Machine Learning

Selecione variáveis numéricas relevantes:

Sugestão:

```python
variaveis = [

"dorms",
"area_util_m2",
"area_total_m2",
"floor",
"total_floors",
"year_built",
"vagas_garagem",
"condominio_brl_monthly",
"iptu_brl_annual",
"price_brl"

]
```

Realize:

- Seleção das variáveis;
- Remoção dos registros incompletos;
- Organização do conjunto final.


Explique:

Por que é necessário realizar a limpeza dos dados antes de treinar um modelo?


---

# 8. Clusterização com K-Means

O objetivo desta etapa é encontrar grupos de imóveis semelhantes.

Utilize as características:

- Número de dormitórios;
- Área;
- Número de vagas;
- Andar;
- Custos do imóvel.


Antes da aplicação do algoritmo:

Realize a normalização dos dados utilizando:

```python
StandardScaler()
```


---

## 8.1 Escolha do número de clusters

Utilize o método do cotovelo:

```python
KMeans()
```

Analise o gráfico obtido e escolha um valor adequado para o número de grupos.


Responda:

1. Quantos clusters foram escolhidos?
2. Qual a justificativa para essa escolha?
3. Quais características diferenciam cada grupo encontrado?


---

## 8.2 Visualização dos clusters

Crie um gráfico utilizando:

Eixo X:

```
area_util_m2
```

Eixo Y:

```
price_brl
```

Utilize as cores para representar os clusters.


Interprete:

- Qual cluster possui imóveis maiores?
- Qual cluster possui imóveis mais caros?
- Existem grupos bem definidos?


---

# 9. Modelo preditivo utilizando Árvore de Decisão

Nesta etapa será desenvolvido um modelo para prever:

```
price_brl
```

utilizando características do imóvel.


Variáveis de entrada:

- Área;
- Quartos;
- Garagem;
- Andar;
- Custos adicionais.


Divida os dados em:

- Treinamento: 80%;
- Teste: 20%.


Utilize:

```python
train_test_split()
```

---

# 10. Treinamento do modelo

Crie uma Árvore de Decisão:

```python
DecisionTreeRegressor(
    max_depth=5
)
```

Treine o modelo utilizando os dados de treinamento.


---

# 11. Avaliação do modelo

Avalie os resultados utilizando:

## Erro Médio Absoluto (MAE)

Mede o erro médio entre o valor real e o valor previsto.


## Coeficiente R²

Indica quanto o modelo consegue explicar os dados.


Analise:

1. O modelo apresenta bom desempenho?
2. O erro obtido é aceitável?
3. Quais fatores podem melhorar a previsão?


---

# 12. Importância das variáveis

Utilize:

```python
arvore.feature_importances_
```

para identificar quais características possuem maior influência no preço.


Responda:

1. Qual variável mais influencia o valor do imóvel?
2. A área possui maior influência que outros fatores?
3. Os resultados fazem sentido considerando o mercado imobiliário?


---

# 13. Desafio adicional

Escolha uma das atividades abaixo:

## Opção 1 - Comparação de modelos

Compare a Árvore de Decisão com:

- KNN;
- Regressão Linear;
- Random Forest.


Compare:

- MAE;
- R².


---

## Opção 2 - Inclusão de variáveis categóricas

Utilize informações como:

- Bairro;
- Zona;
- Tipo de imóvel;
- Condição do imóvel.


Realize a transformação dessas variáveis utilizando:

- Label Encoding;
- One Hot Encoding.


Avalie se o desempenho melhora.


---

## Opção 3 - Análise de mercado

Utilize os dados para responder:

- Quais bairros possuem imóveis mais caros?
- Qual região apresenta maior preço por metro quadrado?
- Existe relação entre proximidade de transporte público e preço?


---

# Entrega

O aluno deverá entregar ao longo do NF (via fork de repo):

1. Notebook `.ipynb` contendo todo o código;
2. Gráficos gerados;
3. Respostas das análises solicitadas;
4. Interpretação dos resultados obtidos pelos modelos.

A avaliação considerará:

- Organização do código;
- Correção da análise;
- Qualidade das visualizações;
- Justificativa das decisões tomadas;
- Interpretação dos resultados de Machine Learning.
