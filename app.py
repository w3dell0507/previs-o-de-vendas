"""
Aplicação Web de Previsão de Vendas com Streamlit e TensorFlow.
"""

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# Configuração da página no Streamlit
st.set_page_config(
    page_title="Previsão de Vendas",
    page_icon="📈",
    layout="wide"
)

# 1. Carregamento dos dados sintéticos
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

# 2. Treinamento do modelo TensorFlow
@st.cache_resource
def treinar_modelo(x_dados: np.ndarray, y_dados: np.ndarray, epocas: int = 300):
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[1])
    ])
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss="mean_squared_error"
    )
    modelo.fit(x_dados, y_dados, epochs=epocas, verbose=0)
    return modelo

# --- INTERFACE STREAMLIT ---
st.title("📈 Painel de Previsão de Vendas com TensorFlow")
st.write("Aplicação para análise exploratória e estimativa de vendas baseada em investimento em marketing.")

df_vendas = carregar_dados()

# Divisão da tela em colunas
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Visualização dos Dados Brutos")
    st.dataframe(df_vendas, use_container_width=True)

with col2:
    st.subheader("📋 Estatísticas Descritivas")
    st.dataframe(df_vendas.describe(), use_container_width=True)

st.divider()

# Treinamento do Modelo
x_dados = df_vendas["investimento_marketing_k"].values.astype(float)
y_dados = df_vendas["vendas_unidades"].values.astype(float)

with st.spinner("Treinando modelo TensorFlow..."):
    modelo = treinar_modelo(x_dados, y_dados)

# Área de Previsão Interativa
st.subheader("🤖 Fazer uma Nova Previsão")

investimento_input = st.slider(
    "Selecione o valor de investimento em marketing (em milhares R$):",
    min_value=0.5,
    max_value=15.0,
    value=7.5,
    step=0.5
)

if st.button("Calcular Previsão"):
    entrada = np.array([[investimento_input]], dtype=float)
    predicao = modelo.predict(entrada, verbose=0)[0][0]
    
    st.success(f"**Previsão de Vendas:** {predicao:.2f} unidades para um investimento de **R$ {investimento_input:.2f}k**")

    # Gráfico simples comparando o histórico com a previsão
    st.line_chart(df_vendas.set_index("investimento_marketing_k")["vendas_unidades"])