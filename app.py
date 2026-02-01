import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================
st.set_page_config(
    page_title="Bank Marketing EDA",
    layout="wide"
)

# =====================================================
# CLASE ANALIZADOR
# =====================================================
class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def get_variable_types(self):
        num_vars = self.df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_vars = self.df.select_dtypes(include=["object"]).columns.tolist()
        return num_vars, cat_vars

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("Menú")

section = st.sidebar.radio(
    "Navegar en:",
    ["Home", "Carga de Dataset", "EDA"],
    key="menu_radio"
)

# =====================================================
# HOME
# =====================================================
if section == "Home":
    st.title("Análisis Exploratorio de Datos – Bank Marketing")

    st.markdown("""
    **Objetivo:**  
    Analizar los datos de la campaña de marketing bancario para identificar
    patrones y factores asociados a la aceptación del producto.
    """)

    st.subheader("Autor")
    st.write("""
    - **Nombre:** Paulo Daniel Liu Cáceda  
    - **Curso:** Especialización en Python for Analytics  
    - **Año:** 2026
    """)

    st.subheader("Tecnologías")
    st.write("Python, Pandas, NumPy, Streamlit, Matplotlib, Seaborn")

# =====================================================
# CARGA DE DATASET
# =====================================================
elif section == "Carga de Dataset":

    st.title("Carga de Dataset")

    use_default = st.checkbox(
        "Usar dataset por defecto",
        value=True,
        key="use_default_dataset"
    )

    if use_default:
        try:
            df = pd.read_csv("Data/BankMarketing.csv", sep=";")
            st.success("Dataset por defecto cargado correctamente")
            st.session_state["df"] = df
            st.dataframe(df.head())
            st.write("Dimensiones:", df.shape)
        except FileNotFoundError:
            st.error("No se encontró el dataset por defecto")

    else:
        uploaded_file = st.file_uploader(
            "Sube tu dataset (CSV)",
            type=["csv"],
            key="file_uploader"
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset cargado correctamente")
            st.session_state["df"] = df
            st.dataframe(df.head())
            st.write("Dimensiones:", df.shape)
        else:
            st.info("Esperando que subas un archivo CSV")

# =====================================================
# EDA
# =====================================================
elif section == "EDA":

    st.title("Exploratory Data Analysis (EDA)")

    if "df" not in st.session_state:
        st.error("Primero debe cargar el dataset en la sección 'Carga de Dataset'")
        st.stop()

    df = st.session_state["df"]
    analyzer = DataAnalyzer(df)

    num_vars, cat_vars = analyzer.get_variable_types()

    tabs = st.tabs([
        "1. Información del Dataset",
        "2. Clasificación de Variables",
        "3. Estadísticas Descriptivas",
        "4. Valores Faltantes",
        "5. Distribución Numérica",
        "6. Variables Categóricas",
        "7. Numérico vs Categórico",
        "8. Categórico vs Categórico",
        "9. Análisis por Parámetros",
        "10. Hallazgos Clave"
    ])

    # =================================================
    # TAB 1 – Información del Dataset
    # =================================================
    with tabs[0]:
        st.subheader("Información general")

        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.write("Dimensiones:", df.shape)
        st.dataframe(df.head())
        st.write("Total de valores nulos:", df.isnull().sum().sum())

    # =================================================
    # TAB 2 – Clasificación de Variables
    # =================================================
    with tabs[1]:
        st.subheader("Clasificación de variables")
        st.write("Variables numéricas:", num_vars)
        st.write("Variables categóricas:", cat_vars)

    # =================================================
    # TAB 3 – Estadísticas descriptivas
    # =================================================
    with tabs[2]:
        st.subheader("Estadísticas descriptivas")

        numeric_cols = df.select_dtypes(include="number").columns

        stats = pd.DataFrame({
            "mean": df[numeric_cols].mean(),
            "median": df[numeric_cols].median(),
            "std": df[numeric_cols].std(),
            "min": df[numeric_cols].min(),
            "max": df[numeric_cols].max(),
            "missing": df[numeric_cols].isnull().sum()
        })

        st.dataframe(stats.round(2))

        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Boxplot de {col}")
            st.pyplot(fig)

    # =================================================
    # TAB 4 – Valores faltantes
    # =================================================
    with tabs[3]:
        st.subheader("Valores faltantes por variable")
        st.dataframe(df.isnull().sum())

    # =================================================
    # TAB 5 – Distribución numérica
    # =================================================
    with tabs[4]:
        st.subheader("Distribución de variables numéricas")

        for col in num_vars:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(col)
            st.pyplot(fig)

    # =================================================
    # TAB 6 – Variables categóricas
    # =================================================
    with tabs[5]:
        st.subheader("Variables categóricas")

        for col in cat_vars:
            counts = df[col].value_counts(dropna=False)
            st.write(col)
            st.dataframe(counts)
            st.bar_chart(counts)

    # =================================================
    # TAB 7 – Numérico vs Categórico
    # =================================================
    with tabs[6]:
        st.subheader("Numérico vs Categórico")

        if num_vars and cat_vars:
            num = st.selectbox(
                "Variable numérica",
                num_vars,
                key="num_vs_cat_num"
            )
            cat = st.selectbox(
                "Variable categórica",
                cat_vars,
                key="num_vs_cat_cat"
            )

            fig, ax = plt.subplots()
            sns.boxplot(x=df[cat], y=df[num], ax=ax)
            st.pyplot(fig)
        else:
            st.info("No hay suficientes variables")

    # =================================================
    # TAB 8 – Categórico vs Categórico
    # =================================================
    with tabs[7]:
        st.subheader("Categórico vs Categórico")

        if len(cat_vars) >= 2:
            cat1 = st.selectbox(
                "Primera variable",
                cat_vars,
                key="cat_cat_1"
            )
            cat2 = st.selectbox(
                "Segunda variable",
                cat_vars,
                index=1,
                key="cat_cat_2"
            )

            ct = pd.crosstab(df[cat1], df[cat2])
            st.dataframe(ct)
        else:
            st.info("No hay suficientes variables categóricas")

    # =================================================
    # TAB 9 – Análisis por parámetros
    # =================================================
    with tabs[8]:
        st.subheader("Análisis basado en parámetros")

        if num_vars:
            var = st.selectbox(
                "Variable numérica",
                num_vars,
                key="filter_numeric"
            )

            min_val, max_val = float(df[var].min()), float(df[var].max())

            rango = st.slider(
                "Selecciona rango",
                min_val,
                max_val,
                (min_val, max_val),
                key="range_slider"
            )

            st.dataframe(df[(df[var] >= rango[0]) & (df[var] <= rango[1])])
        else:
            st.info("No hay variables numéricas")

    # =================================================
    # TAB 10 – Hallazgos
    # =================================================
    with tabs[9]:
        st.subheader("Hallazgos clave")
        st.markdown("""
        - Identificar variables con mayor impacto.
        - Detectar patrones relevantes.
        - Evaluar calidad de datos.
        - Proponer mejoras para futuras campañas.
        """)
