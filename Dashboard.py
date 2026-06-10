import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Diabetes Dashboard", layout="wide")

st.markdown("""
    <style>
        .block-container { padding: 2rem 2.5rem; }
        h1 { font-size: 1.6rem; font-weight: 600; }
        h2 { font-size: 1.1rem; font-weight: 500; color: #555; }
        .stMetric label { font-size: 0.8rem; color: #888; }
    </style>
""", unsafe_allow_html=True) 


# ── Carregamento e cache ──────────────────────────────────────────────
@st.cache_data
def carregar_dados():
    df = pd.read_csv("diabetes_limpo.csv")
    return df

@st.cache_data
def treinar_modelos(df):
    df_m = df.copy()
    le_g = LabelEncoder()
    le_s = LabelEncoder()
    df_m["gender"]          = le_g.fit_transform(df_m["gender"])
    df_m["smoking_history"] = le_s.fit_transform(df_m["smoking_history"])

    X = df_m[["age","bmi","HbA1c_level","blood_glucose_level",
               "gender","smoking_history","hypertension","heart_disease"]]
    y = df_m["diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(X_train_s, y_train)

    y_pred_dt  = dt.predict(X_test)
    y_pred_knn = knn.predict(X_test_s)

    return dt, knn, scaler, y_test, y_pred_dt, y_pred_knn, le_g, le_s

df = carregar_dados()
dt, knn, scaler, y_test, y_pred_dt, y_pred_knn, le_g, le_s = treinar_modelos(df)


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — Filtros
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Filtros")

    genero = st.multiselect(
        "Gênero",
        options=df["gender"].unique().tolist(),
        default=df["gender"].unique().tolist()
    )

    idade_range = st.slider(
        "Faixa etária",
        int(df["age"].min()), int(df["age"].max()),
        (int(df["age"].min()), int(df["age"].max()))
    )

    smoking = st.multiselect(
        "Histórico de tabagismo",
        options=df["smoking_history"].unique().tolist(),
        default=df["smoking_history"].unique().tolist()
    )

    hiper = st.selectbox("Hipertensão", ["Todos", "Sim", "Não"])
    card  = st.selectbox("Doença cardíaca", ["Todos", "Sim", "Não"])

# Aplicar filtros
df_f = df[
    df["gender"].isin(genero) &
    df["age"].between(idade_range[0], idade_range[1]) &
    df["smoking_history"].isin(smoking)
]
if hiper == "Sim":   df_f = df_f[df_f["hypertension"] == 1]
elif hiper == "Não": df_f = df_f[df_f["hypertension"] == 0]
if card == "Sim":    df_f = df_f[df_f["heart_disease"] == 1]
elif card == "Não":  df_f = df_f[df_f["heart_disease"] == 0]


# ══════════════════════════════════════════════════════════════════════
# ABAS
# ══════════════════════════════════════════════════════════════════════
aba1, aba2, aba3 = st.tabs(["Análise dos Dados", "Teorema de Bayes", "Classificação"])


# ══════════════════════════════════════════════════════════════════════
# ABA 1 — EDA
# ══════════════════════════════════════════════════════════════════════
with aba1:
    st.title("Análise Exploratória")

    # Métricas rápidas
    total     = len(df_f)
    diab      = df_f["diabetes"].sum()
    pct_diab  = diab / total * 100 if total > 0 else 0
    idade_med = df_f["age"].median()
    bmi_med   = df_f["bmi"].median()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pacientes", f"{total:,}")
    c2.metric("Diabéticos", f"{diab:,}")
    c3.metric("Taxa de diabetes", f"{pct_diab:.1f}%")
    c4.metric("Idade mediana", f"{idade_med:.0f} anos")

    st.divider()

    # Linha 1 — distribuição alvo + distribuições numéricas
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Distribuição da variável alvo**")
        contagem = df_f["diabetes"].value_counts().reset_index()
        contagem.columns = ["Classe", "Contagem"]
        contagem["Classe"] = contagem["Classe"].map({0: "Não diabético", 1: "Diabético"})
        fig = px.bar(contagem, x="Classe", y="Contagem",
                     color="Classe",
                     color_discrete_map={"Não diabético": "#4CAF50", "Diabético": "#F44336"},
                     text="Contagem")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=300,
                          margin=dict(t=10, b=10), plot_bgcolor="white",
                          yaxis=dict(showgrid=False), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Distribuição das variáveis numéricas por diagnóstico**")
        var_sel = st.selectbox("Variável", ["age", "bmi", "HbA1c_level", "blood_glucose_level"],
                               key="dist_var")
        fig = px.histogram(df_f, x=var_sel, color=df_f["diabetes"].map({0:"Não diabético",1:"Diabético"}),
                           barmode="overlay", opacity=0.7, nbins=40,
                           color_discrete_map={"Não diabético": "#4CAF50", "Diabético": "#F44336"},
                           labels={"color": ""})
        fig.update_layout(height=300, margin=dict(t=10, b=10),
                          plot_bgcolor="white", legend=dict(orientation="h", y=1.1),
                          xaxis_title=var_sel, yaxis_title="Frequência")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Linha 2 — taxa por faixa etária + taxa por BMI
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Taxa de diabetes por faixa etária**")
        df_f2 = df_f.copy()
        df_f2["faixa_etaria"] = pd.cut(df_f2["age"],
            bins=[0,18,30,45,60,80],
            labels=["< 18","18–30","31–45","46–60","> 60"])
        taxa_idade = (df_f2.groupby("faixa_etaria", observed=True)["diabetes"]
                          .mean() * 100).reset_index()
        taxa_idade.columns = ["Faixa", "Taxa (%)"]
        fig = px.bar(taxa_idade, x="Faixa", y="Taxa (%)",
                     text=taxa_idade["Taxa (%)"].apply(lambda x: f"{x:.1f}%"),
                     color_discrete_sequence=["#5C9BD6"])
        fig.update_traces(textposition="outside")
        fig.update_layout(height=300, margin=dict(t=10, b=10),
                          plot_bgcolor="white", yaxis=dict(showgrid=False),
                          xaxis_title="", yaxis_title="% diabéticos")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("**Taxa de diabetes por faixa de IMC**")
        df_f2["faixa_bmi"] = pd.cut(df_f2["bmi"],
            bins=[0,18.5,25,30,40,100],
            labels=["Abaixo\npeso","Normal","Sobrepeso","Obeso","Ob.\nMórbida"])
        taxa_bmi = (df_f2.groupby("faixa_bmi", observed=True)["diabetes"]
                        .mean() * 100).reset_index()
        taxa_bmi.columns = ["Faixa", "Taxa (%)"]
        fig = px.bar(taxa_bmi, x="Faixa", y="Taxa (%)",
                     text=taxa_bmi["Taxa (%)"].apply(lambda x: f"{x:.1f}%"),
                     color_discrete_sequence=["#E8915A"])
        fig.update_traces(textposition="outside")
        fig.update_layout(height=300, margin=dict(t=10, b=10),
                          plot_bgcolor="white", yaxis=dict(showgrid=False),
                          xaxis_title="", yaxis_title="% diabéticos")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Linha 3 — comorbidades + mapa de correlação
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("**Comorbidades e risco de diabetes**")
        dados_comor = pd.DataFrame({
            "Condição": ["Sem hipertensão","Com hipertensão","Sem doença cardíaca","Com doença cardíaca"],
            "Taxa (%)": [
                df_f[df_f["hypertension"]==0]["diabetes"].mean()*100,
                df_f[df_f["hypertension"]==1]["diabetes"].mean()*100,
                df_f[df_f["heart_disease"]==0]["diabetes"].mean()*100,
                df_f[df_f["heart_disease"]==1]["diabetes"].mean()*100,
            ],
            "Grupo": ["Hipertensão","Hipertensão","Doença cardíaca","Doença cardíaca"]
        })
        fig = px.bar(dados_comor, x="Condição", y="Taxa (%)", color="Grupo",
                     text=dados_comor["Taxa (%)"].apply(lambda x: f"{x:.1f}%"),
                     color_discrete_sequence=["#5C9BD6","#D65C5C"])
        fig.update_traces(textposition="outside")
        fig.update_layout(height=300, margin=dict(t=10, b=10),
                          plot_bgcolor="white", yaxis=dict(showgrid=False),
                          xaxis_title="", yaxis_title="% diabéticos",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.markdown("**Mapa de correlação**")
        cols_corr = ["age","bmi","HbA1c_level","blood_glucose_level",
                     "hypertension","heart_disease","diabetes"]
        corr = df_f[cols_corr].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# ABA 2 — TEOREMA DE BAYES
# ══════════════════════════════════════════════════════════════════════
with aba2:
    st.title("Teorema de Bayes")

    st.markdown("""
    O Teorema de Bayes calcula a probabilidade de um paciente ser diabético
    com base em uma evidência observada — neste projeto, o **nível de glicose no sangue**.

    A fórmula aplicada é:

    > **P(diabético | glicose alta) = P(glicose alta | diabético) × P(diabético) / P(glicose alta)**

    - **P(C)** — probabilidade a priori: proporção de diabéticos no dataset, sem considerar nenhuma informação do paciente  
    - **P(X|C)** — verossimilhança: dado que o paciente é diabético, qual a chance de ter glicose alta?  
    - **P(C|X)** — probabilidade a posteriori: após observar a glicose, qual a probabilidade de ser diabético?
    """)

    st.divider()

    limiar = st.slider("Limiar de glicose alta (mg/dL)", 100, 200, 126, step=1)

    df_b = df_f.copy()
    df_b["glicose_alta"] = (df_b["blood_glucose_level"] >= limiar).astype(int)

    total_b         = len(df_b)
    p_diab          = df_b["diabetes"].mean()
    p_nao_diab      = 1 - p_diab
    p_gli_alta      = df_b["glicose_alta"].mean()
    p_gli_norm      = 1 - p_gli_alta

    p_gli_alta_diab     = df_b[df_b["diabetes"]==1]["glicose_alta"].mean()
    p_gli_alta_nao_diab = df_b[df_b["diabetes"]==0]["glicose_alta"].mean()
    p_gli_norm_diab     = 1 - p_gli_alta_diab
    p_gli_norm_nao_diab = 1 - p_gli_alta_nao_diab

    # Posteriori
    p_post_alta = (p_gli_alta_diab * p_diab) / p_gli_alta if p_gli_alta > 0 else 0
    p_post_norm = (p_gli_norm_diab * p_diab) / p_gli_norm if p_gli_norm > 0 else 0

    # Métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("P(diabético) — a priori",        f"{p_diab*100:.2f}%")
    col2.metric("P(diabético | glicose ALTA)",     f"{p_post_alta*100:.2f}%",
                delta=f"+{(p_post_alta - p_diab)*100:.2f}pp")
    col3.metric("P(diabético | glicose NORMAL)",   f"{p_post_norm*100:.2f}%",
                delta=f"{(p_post_norm - p_diab)*100:.2f}pp")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Atualização bayesiana da probabilidade**")
        fig = go.Figure()
        categorias = ["A priori\n(sem info)", f"Glicose ≥ {limiar}\n(alta)", f"Glicose < {limiar}\n(normal)"]
        valores    = [p_diab*100, p_post_alta*100, p_post_norm*100]
        cores      = ["#888888", "#F44336", "#4CAF50"]
        fig.add_trace(go.Bar(x=categorias, y=valores,
                             marker_color=cores,
                             text=[f"{v:.1f}%" for v in valores],
                             textposition="outside"))
        fig.update_layout(height=320, showlegend=False,
                          plot_bgcolor="white", yaxis=dict(showgrid=False),
                          margin=dict(t=20, b=10),
                          yaxis_title="P(diabético) %")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Verossimilhanças P(glicose | classe)**")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Diabético",
                             x=["Glicose alta","Glicose normal"],
                             y=[p_gli_alta_diab*100, p_gli_norm_diab*100],
                             marker_color="#F44336",
                             text=[f"{p_gli_alta_diab*100:.1f}%", f"{p_gli_norm_diab*100:.1f}%"],
                             textposition="outside"))
        fig.add_trace(go.Bar(name="Não diabético",
                             x=["Glicose alta","Glicose normal"],
                             y=[p_gli_alta_nao_diab*100, p_gli_norm_nao_diab*100],
                             marker_color="#4CAF50",
                             text=[f"{p_gli_alta_nao_diab*100:.1f}%", f"{p_gli_norm_nao_diab*100:.1f}%"],
                             textposition="outside"))
        fig.update_layout(barmode="group", height=320,
                          plot_bgcolor="white", yaxis=dict(showgrid=False),
                          margin=dict(t=20, b=10),
                          legend=dict(orientation="h", y=1.1),
                          yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Cálculo passo a passo**")

    st.markdown(f"""
    | Etapa | Valor |
    |---|---|
    | P(diabético) — a priori | {p_diab:.4f} |
    | P(glicose alta \\| diabético) — verossimilhança | {p_gli_alta_diab:.4f} |
    | P(glicose alta) — evidência | {p_gli_alta:.4f} |
    | **P(diabético \\| glicose alta) — a posteriori** | **{p_post_alta:.4f}** |
    """)


# ══════════════════════════════════════════════════════════════════════
# ABA 3 — CLASSIFICAÇÃO
# ══════════════════════════════════════════════════════════════════════
with aba3:
    st.title("Algoritmos de Classificação")

    # ── Explicações ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Árvore de Decisão")
        st.markdown("""
        Aprende uma sequência de regras **se/então** a partir dos dados.
        Cada nó da árvore faz uma pergunta sobre um atributo (ex: *"glicose > 126?"*),
        e os ramos levam à próxima pergunta até chegar a uma folha com a classe predita.

        - Fácil de interpretar e explicar  
        - Captura combinações entre variáveis  
        - `max_depth=5` evita memorizar o treino (overfitting)
        """)

    with col2:
        st.markdown("##### KNN — K-Nearest Neighbors")
        st.markdown("""
        Para classificar um novo paciente, encontra os **11 pacientes mais parecidos**
        no dataset de treino e vota na classe mais frequente entre eles.
        A "parecença" é medida por distância euclidiana entre os atributos.

        - Intuitivo: casos similares têm diagnósticos similares  
        - Sensível à escala → variáveis normalizadas com StandardScaler  
        - `k=11` (ímpar) evita empate na votação
        """)

    st.divider()

    # ── Métricas de desempenho ────────────────────────────────────────
    st.markdown("##### Desempenho no conjunto de teste")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    def metricas(y_true, y_pred):
        return {
            "Acurácia":  round(accuracy_score(y_true, y_pred), 4),
            "Precisão":  round(precision_score(y_true, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "F1-Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        }

    met_dt  = metricas(y_test, y_pred_dt)
    met_knn = metricas(y_test, y_pred_knn)

    df_met = pd.DataFrame([met_dt, met_knn], index=["Árvore de Decisão", "KNN"]).reset_index()
    df_met.columns = ["Modelo","Acurácia","Precisão","Recall","F1-Score"]

    fig = go.Figure()
    cores_met = {"Acurácia":"#5C9BD6","Precisão":"#E8915A","Recall":"#4CAF50","F1-Score":"#9C6DD6"}
    x = ["Árvore de Decisão", "KNN"]
    for metrica, cor in cores_met.items():
        fig.add_trace(go.Bar(name=metrica, x=x,
                             y=[met_dt[metrica], met_knn[metrica]],
                             marker_color=cor,
                             text=[f"{met_dt[metrica]:.3f}", f"{met_knn[metrica]:.3f}"],
                             textposition="outside"))
    fig.update_layout(barmode="group", height=320,
                      plot_bgcolor="white", yaxis=dict(showgrid=False, range=[0,1.15]),
                      margin=dict(t=20, b=10),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Matrizes de confusão ──────────────────────────────────────────
    st.markdown("##### Matrizes de confusão")

    col3, col4 = st.columns(2)

    def plot_cm(y_true, y_pred, titulo):
        cm = confusion_matrix(y_true, y_pred)
        labels = ["Não diabético", "Diabético"]
        fig = px.imshow(cm, text_auto=True,
                        x=[f"Pred: {l}" for l in labels],
                        y=[f"Real: {l}" for l in labels],
                        color_continuous_scale="Blues",
                        title=titulo)
        fig.update_layout(height=300, margin=dict(t=40, b=10),
                          coloraxis_showscale=False)
        return fig

    with col3:
        st.plotly_chart(plot_cm(y_test, y_pred_dt, "Árvore de Decisão"),
                        use_container_width=True)
    with col4:
        st.plotly_chart(plot_cm(y_test, y_pred_knn, "KNN"),
                        use_container_width=True)

    st.divider()

    # ── Classificador interativo ──────────────────────────────────────
    st.markdown("##### Classificar novo paciente")
    st.caption("Preencha os dados e veja a predição dos dois modelos.")

    c1, c2, c3, c4 = st.columns(4)
    idade_p  = c1.number_input("Idade",          2,  80, 45)
    bmi_p    = c2.number_input("IMC",            10.0, 60.0, 27.0, step=0.1)
    hba1c_p  = c3.number_input("HbA1c (%)",      3.5, 9.5, 5.5, step=0.1)
    glicose_p = c4.number_input("Glicose (mg/dL)", 80, 300, 120)

    c5, c6, c7, c8 = st.columns(4)
    genero_p  = c5.selectbox("Gênero",     ["Female","Male"])
    smoking_p = c6.selectbox("Tabagismo",  ["never","current","former","unknown"])
    hiper_p   = c7.selectbox("Hipertensão",["Não","Sim"])
    card_p    = c8.selectbox("D. cardíaca",["Não","Sim"])

    if st.button("Classificar"):
        genero_enc  = le_g.transform([genero_p])[0]
        smoking_enc = le_s.transform([smoking_p])[0]
        hiper_enc   = 1 if hiper_p == "Sim" else 0
        card_enc    = 1 if card_p  == "Sim" else 0

        entrada = np.array([[idade_p, bmi_p, hba1c_p, glicose_p,
                             genero_enc, smoking_enc, hiper_enc, card_enc]])

        pred_dt  = dt.predict(entrada)[0]
        pred_knn = knn.predict(scaler.transform(entrada))[0]

        prob_dt  = dt.predict_proba(entrada)[0][1]
        prob_knn = knn.predict_proba(scaler.transform(entrada))[0][1]

        def badge(pred, prob):
            cor   = "#F44336" if pred == 1 else "#4CAF50"
            label = "Diabético" if pred == 1 else "Não diabético"
            return f'<div style="background:{cor};color:white;padding:10px 16px;border-radius:8px;font-size:1rem;font-weight:500">{label} ({prob*100:.1f}%)</div>'

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Árvore de Decisão**")
            st.markdown(badge(pred_dt, prob_dt), unsafe_allow_html=True)
        with r2:
            st.markdown("**KNN**")
            st.markdown(badge(pred_knn, prob_knn), unsafe_allow_html=True)