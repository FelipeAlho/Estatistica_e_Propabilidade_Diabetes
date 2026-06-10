# Diabetes Prediction Dashboard

Projeto desenvolvido para a disciplina de **Estatística e Probabilidade**, com foco em análise exploratória, tratamento de dados, aplicação do **Teorema de Bayes**, modelos de classificação e construção de um dashboard interativo.

## Sobre o projeto

O objetivo é analisar um conjunto de dados sobre risco de diabetes e permitir que o usuário visualize padrões importantes, calcule probabilidades e teste classificações para novos pacientes.

O projeto contém:

- limpeza e tratamento do dataset;
- análise exploratória dos dados;
- aplicação do Teorema de Bayes;
- treinamento de dois algoritmos de classificação;
- dashboard interativo em Streamlit.

## Dataset utilizado

O dataset foi obtido no Kaggle:

https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

O arquivo original utilizado foi:

```text
diabetes_prediction_dataset.csv
```

Após o tratamento dos dados, foi gerado o arquivo:

```text
diabetes_limpo.csv
```

## Variável alvo

A variável alvo do projeto é:

```text
diabetes
```

Ela indica se o paciente possui ou não diabetes:

- `0`: não diabético
- `1`: diabético

## Variáveis utilizadas

As principais variáveis analisadas foram:

- `gender`: gênero;
- `age`: idade;
- `hypertension`: presença de hipertensão;
- `heart_disease`: presença de doença cardíaca;
- `smoking_history`: histórico de tabagismo;
- `bmi`: índice de massa corporal;
- `HbA1c_level`: nível de hemoglobina glicada;
- `blood_glucose_level`: nível de glicose no sangue;
- `diabetes`: diagnóstico de diabetes.

## Tratamentos realizados

Durante a limpeza dos dados, foram aplicados os seguintes tratamentos:

- remoção de registros duplicados;
- remoção da categoria `Other` em `gender`, por representar uma proporção muito pequena;
- padronização da variável `smoking_history`;
- remoção de idades menores que 2 anos;
- tratamento de outliers no `bmi` por winsorização entre os percentis 1% e 99%;
- criação de variáveis derivadas, como faixa etária, faixa de IMC e glicose alta.

Esses tratamentos foram feitos para melhorar a qualidade dos dados e reduzir distorções nas análises e nos modelos.

## Teorema de Bayes

A análise bayesiana foi aplicada para estimar a probabilidade de um paciente ser diabético a partir da evidência de glicose alta.

Foi utilizada a fórmula:

```text
P(diabético | glicose alta) = P(glicose alta | diabético) × P(diabético) / P(glicose alta)
```

No dashboard, o usuário pode alterar o limiar de glicose alta e observar como a probabilidade posterior muda.

## Modelos de classificação

Foram utilizados dois algoritmos de classificação:

1. **Árvore de Decisão**
   - modelo baseado em regras do tipo “se/então”;
   - configurado com profundidade máxima igual a 5.

2. **KNN — K-Nearest Neighbors**
   - classifica novos casos com base nos 11 vizinhos mais próximos;
   - utiliza normalização dos dados com `StandardScaler`.

Os modelos são avaliados por métricas como:

- acurácia;
- precisão;
- recall;
- F1-score;
- matriz de confusão.

## Dashboard

O dashboard foi construído com **Streamlit** e possui três abas principais:

1. **Análise dos Dados**
   - gráficos de distribuição;
   - taxas de diabetes por idade, IMC e comorbidades;
   - mapa de correlação.

2. **Teorema de Bayes**
   - cálculo das probabilidades a priori, verossimilhança e posteriori;
   - visualização da atualização bayesiana.

3. **Classificação**
   - comparação entre Árvore de Decisão e KNN;
   - métricas de desempenho;
   - formulário para classificar um novo paciente.

## Estrutura do projeto

```text
Super_Projeto/
│
├── Analise.ipynb
├── Dashboard.py
├── diabetes_prediction_dataset.csv
├── diabetes_limpo.csv
└── README.md
```

## Como executar o projeto

### 1. Clonar o repositório

```bash
git clone LINK_DO_REPOSITORIO
cd Super_Projeto
```

### 2. Instalar as dependências

```bash
pip install pandas numpy matplotlib seaborn streamlit plotly scikit-learn jupyter
```

### 3. Executar o dashboard

```bash
streamlit run Dashboard.py
```

Depois disso, o Streamlit abrirá o dashboard no navegador.

## Integrantes

- Felipe Menezes Alho
- Pedro Mansour Sales
- Josiel Teixeira Costa
