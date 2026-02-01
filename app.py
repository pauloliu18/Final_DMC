import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================
st.set_page_config(page_title="Bank Marketing EDA", layout="wide")

DEFAULT_DATASET_PATH = "data/bank_marketing.csv"

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

    def descriptive_stats(self):
        return self.df.describe()

    def missing_values(self):
        return self.df.isnull().sum()

    def plot_numeric_distribution(self, column):
        fig, ax = plt.subplots()
        sns.histplot(self.df[column], kde=True, ax=ax)
        ax.set_title(f"Distribución de {column}")
        return fig

    def categorical_counts(self, column):
        return self.df[column].value_counts()

# =====================================================
# SIDEBAR - MENÚ
# =====================================================
st.sidebar.title("Menú")

section = st.sidebar.radio(
    "Navegar en:",
    ["Home", "Carga de Dataset", "EDA"]
)

# =====================================================
# HOME
# =====================================================
if section == "Home":
    st.title("Análisis Exploratorio de Datos – Bank Marketing")

    st.markdown("""
    **Objetivo:**  
    Analizar los datos de la última campaña de marketing bancario para identificar
    patrones y factores asociados a la aceptación del producto.
    """)

    st.subheader("Autor")
    st.write("""
    - **Nombre:** Paulo Daniel Liu Cáceda  
    - **Curso:** Especialización en Python for Analytics  
    - **Año:** 2026
    """)

    st.subheader("Dataset")
    st.write("""
    Dataset correspondiente a campañas de marketing de una institución financiera,
    cuyo objetivo es entender qué variables influyen en la aceptación del producto.
    """)

    st.subheader("Tecnologías")
    st.write("Python, Pandas, NumPy, Streamlit, Matplotlib, Seaborn")

# =====================================================
# CARGA DE DATASET
# =====================================================
elif section == "Carga de Dataset":

    st.title("Carga de Dataset")

    use_default = st.checkbox("Usar dataset por defecto", value=True)

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
        uploaded_file = st.file_uploader("Sube tu dataset (CSV)", type=["csv"])
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
        "3. Estadísticas descriptivas",
        "4. Análisis de valores faltantes",
        "5. Distribución de variables numéricas",
        "6. Análisis de variables categóricas",
        "7. Análisis bivariado (numérico vs categórico)",
        "8. Análisis bivariado (categórico vs categórico)",
        "9. Análisis basado en parámetros",
        "10. Hallazgos clave"
    ])

    # 1️⃣ Información del Dataset
    with tabs[0]:
        df
        df.info()
        st.subheader("Información general")
        st.write("Dimensiones:", df.shape)
        st.dataframe(df.head())
        st.write("Tipos de datos")
        st.dataframe(df.dtypes)
        # Resumen rápido de nulos totales
        st.write("Total de valores nulos en el dataset:", df.isnull().sum().sum())

    # 2️⃣ Clasificación de Variables
    with tabs[1]:
            st.subheader("Clasificación de variables")
            st.write("Variables numéricas:", num_vars)
            st.write("Variables categóricas:", cat_vars)

            st.divider()

            st.subheader("Conteo de variables categóricas")
            for col in cat_vars:
                st.markdown(f"### {col}")
                conteo = df[col].value_counts().reset_index()
                conteo.columns = [col, "Frecuencia"]
                st.dataframe(conteo)

            st.divider()

            st.subheader("Conteo de valores únicos (numéricas)")
            conteo_num = pd.DataFrame({
            "Variable": num_vars,
            "Valores únicos": [df[col].nunique() for col in num_vars]
            })
            st.dataframe(conteo_num)
      

# 3️⃣ Estadísticas descriptivas con media, mediana y dispersión
    with tabs[2]:
        st.subheader("Estadísticas descriptivas detalladas")

        numeric_cols = df.select_dtypes(include='number').columns
        st.write("🔹 Variables numéricas")
        
        # Crear dataframe con estadísticas clave
        stats = pd.DataFrame({
            'count': df[numeric_cols].count(),
            'mean': df[numeric_cols].mean(),
            'median': df[numeric_cols].median(),
            'std_dev': df[numeric_cols].std(),
            'variance': df[numeric_cols].var(),
            'min': df[numeric_cols].min(),
            'max': df[numeric_cols].max(),
            'range': df[numeric_cols].max() - df[numeric_cols].min(),
            '25%': df[numeric_cols].quantile(0.25),
            '50%': df[numeric_cols].quantile(0.5),
            '75%': df[numeric_cols].quantile(0.75),
            'iqr': df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25),
            'missing': df[numeric_cols].isnull().sum()
        })
        
        st.dataframe(stats.round(2))  # redondear a 2 decimales

        # 🔹 Boxplots para visualización de dispersión y outliers
        st.write("🔹 Visualización de dispersión y outliers")
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Boxplot de {col}")
            st.pyplot(fig)


    # 4 Conteo de categorías y visualización
    with tabs[3]:
        st.subheader("Conteo de categorías y visualización")

    categorical_cols = df.select_dtypes(include='object').columns

    for col in categorical_cols:
        st.write(f"🔹 Columna: {col}")
        
        # Conteo de valores
        counts = df[col].value_counts(dropna=False)
        st.dataframe(counts)
        
        # Gráfico de barras simple
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6,3))
        counts.plot(kind='bar', color='lightgreen', ax=ax)
        ax.set_title(f"Distribución de {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)
        
        # Discusión breve automática
        st.write("**Discusión breve:**")
        total = counts.sum()
        top_category = counts.idxmax()
        top_freq = counts.max()
        missing = df[col].isnull().sum()
        st.write(f"- La categoría más frecuente es '{top_category}' con {top_freq} observaciones ({top_freq/total*100:.2f}%).")
        st.write(f"- Total de valores únicos: {df[col].nunique()}")
        st.write(f"- Valores faltantes: {missing} ({missing/total*100:.2f}%)")
        st.write("---")




    # 5 Histogramas y visualización de distribución
    with tabs[4]:
     st.subheader("Histogramas y distribución de variables numéricas")
    
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(df[col], bins=20, kde=True, color='skyblue', ax=ax)
        ax.set_title(f"Histograma de {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
        
        st.pyplot(fig)
        
        # Interpretación rápida
        st.write(f"**Interpretación de {col}:**")
        st.write(f"- Media: {df[col].mean():.2f}, Mediana: {df[col].median():.2f}")
        st.write(f"- Desviación estándar: {df[col].std():.2f}")
        st.write(f"- La distribución parece {'asimétrica a la derecha' if df[col].skew() > 0 else 'asimétrica a la izquierda' if df[col].skew() < 0 else 'simétrica'}")
        st.write("---")

    # 6️⃣ Conteos, proporciones y gráficos de barras
    with tabs[4]:
        st.subheader("Conteos, proporciones y gráficos de barras")

        categorical_cols = df.select_dtypes(include='object').columns

        for col in categorical_cols:
            st.write(f"🔹 Columna: {col}")

            # Conteo de valores
            counts = df[col].value_counts(dropna=False)
            st.dataframe(counts)

            # Proporciones (%)
            proportions = (counts / counts.sum() * 100).round(2)
            proportions_df = pd.DataFrame({'Conteo': counts, 'Proporción (%)': proportions})
            st.dataframe(proportions_df)

            # Gráfico de barras
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6,3))
            counts.plot(kind='bar', color='salmon', ax=ax)
            ax.set_title(f"Distribución de {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frecuencia")
            st.pyplot(fig)

            # Discusión breve automática
            st.write("**Interpretación rápida:**")
            st.write(f"- Categoría más frecuente: '{counts.idxmax()}' ({proportions[counts.idxmax()]}%)")
            st.write(f"- Total de categorías: {df[col].nunique()}")
            st.write(f"- Valores faltantes: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.2f}%)")
            st.write("---")

    # 7️⃣ Numérico vs Categórico
    with tabs[6]:
        st.subheader("Numérico vs Categórico")
        if num_vars and cat_vars:
            num = st.selectbox("Variable numérica", num_vars)
            cat = st.selectbox("Variable categórica", cat_vars)
            fig, ax = plt.subplots()
            sns.boxplot(x=df[cat], y=df[num], ax=ax)
            ax.set_title(f"{num} por {cat}")
            st.pyplot(fig)
        else:
            st.info("No hay suficientes variables")

    # 8️⃣ Categórico vs Categórico
    with tabs[7]:
        st.subheader("Categórico vs Categórico")
        if len(cat_vars) >= 2:
            cat1 = st.selectbox("Primera variable", cat_vars)
            cat2 = st.selectbox("Segunda variable", cat_vars, index=1)
            ct = pd.crosstab(df[cat1], df[cat2])
            st.dataframe(ct)
        else:
            st.info("No hay suficientes variables categóricas")

    # 9️⃣ Análisis por parámetros
    with tabs[8]:
        st.subheader("Análisis basado en parámetros")
        if num_vars:
            var = st.selectbox("Variable numérica a filtrar", num_vars)
            min_val, max_val = float(df[var].min()), float(df[var].max())
            rango = st.slider(
                "Selecciona rango",
                min_val,
                max_val,
                (min_val, max_val)
            )
            st.dataframe(df[(df[var] >= rango[0]) & (df[var] <= rango[1])])
        else:
            st.info("No hay variables numéricas")

    # 🔟 Hallazgos clave
    with tabs[9]:
        st.subheader("Hallazgos clave")
        st.markdown("""
        - Identificar variables con mayor influencia en la aceptación del producto.
        - Detectar patrones relevantes en el comportamiento del cliente.
        - Evaluar la calidad de los datos.
        - Proponer recomendaciones para futuras campañas.
        """)
