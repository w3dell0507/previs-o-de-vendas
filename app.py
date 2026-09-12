import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Previsão de Vendas", page_icon="📈", layout="wide")

@st.cache_data
def carregar_dados() -> pd.DataFrame:
    dados = {
        "mes": list(range(1, 13)),
        "investimento_marketing_k": [
            1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5
        ],
        "vendas_unidades": [
            120, 145, 160, 185, 210, 230, 255, 275, 300, 320, 345, 365
        ]
    }
    return pd.DataFrame(dados)

@st.cache_resource
def treinar_modelo(X: np.ndarray, y: np.ndarray):
    modelo = LinearRegression()
    modelo.fit(X, y)
    return modelo

st.title("📈 Painel de Previsão de Vendas")
st.write("Aplicação para análise exploratória e estimativa de vendas.")

df_vendas = carregar_dados()

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📊 Visualização dos Dados Brutos")
    st.dataframe(df_vendas, use_container_width=True)

with col2:
    st.subheader("📋 Estatísticas Descritivas")
    st.dataframe(df_vendas.describe(), use_container_width=True)

st.divider()

X_dados = df_vendas[["investimento_marketing_k"]].values
y_dados = df_vendas["vendas_unidades"].values
modelo = treinar_modelo(X_dados, y_dados)

st.subheader("🤖 Fazer uma Nova Previsão")
investimento_input = st.slider(
    "Selecione o valor de investimento em marketing (em milhares R$):",
    min_value=0.5,
    max_value=15.0,
    value=7.5,
    step=0.5
)

if st.button("Calcular Previsão"):
    predicao = modelo.predict([[investimento_input]])[0]
    st.success(f"**Previsão de Vendas:** {predicao:.2f} unidades para um investimento de **R$ {investimento_input:.2f}k**")
    st.line_chart(df_vendas.set_index("investimento_marketing_k")["vendas_unidades"])
