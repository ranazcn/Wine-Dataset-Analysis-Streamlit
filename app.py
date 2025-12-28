import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------
# SAYFA AYARLARI
# ---------------------------------------------------
st.set_page_config(
    page_title="Wine Dataset Analysis",
    page_icon="🍷",
    layout="wide",
)

st.title("🍷 Wine Data Analysis Dashboard")
st.markdown("UCI Wine veri seti ile Streamlit üzerinden kapsamlı analiz ve makine öğrenmesi uygulaması.")

# ---------------------------------------------------
# VERİYİ YÜKLE
# ---------------------------------------------------
@st.cache_data
def load_wine_data():
    # wine.data ile aynı klasörde olmalı
    df = pd.read_csv("wine.data", header=None)
    df.columns = [
        "Class",
        "Alcohol",
        "Malic_acid",
        "Ash",
        "Alcalinity_of_ash",
        "Magnesium",
        "Total_phenols",
        "Flavanoids",
        "Nonflavanoid_phenols",
        "Proanthocyanins",
        "Color_intensity",
        "Hue",
        "OD280_OD315",
        "Proline",
    ]
    df["Class"] = df["Class"].astype(int)
    return df

df = load_wine_data()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.header("⚙️ Ayarlar")

    analysis_type = st.selectbox(
        "Analiz tipi:",
        [
            "Genel Bakış",
            "Yapısal Bilgiler",
            "Değişken Dağılımları",
            "Korelasyon & Isı Haritası",
            "PCA Analizi",
            "Sınıflandırma (Random Forest)",
            "Dashboard / Özet"
        ],
    )

    feature_for_dist = st.selectbox(
        "Dağılım için sayısal değişken:",
        [
            "Alcohol",
            "Malic_acid",
            "Ash",
            "Alcalinity_of_ash",
            "Magnesium",
            "Total_phenols",
            "Flavanoids",
            "Nonflavanoid_phenols",
            "Proanthocyanins",
            "Color_intensity",
            "Hue",
            "OD280_OD315",
            "Proline",
        ],
    )

    st.markdown("---")
    st.markdown("**Sınıf açıklaması**")
    st.markdown(
        """
        - Class 1: Şarap tipi 1  
        - Class 2: Şarap tipi 2  
        - Class 3: Şarap tipi 3  
        """
    )

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "📋 Genel Bakış",
        "🔍 Yapı",
        "📈 Dağılımlar",
        "📊 Korelasyon",
        "🧠 PCA",
        "🤖 Random Forest",
        "📍 Dashboard",
    ]
)

# ---------------------------------------------------
# TAB 1 – GENEL BAKIŞ
# ---------------------------------------------------
with tab1:
    st.subheader("Verinin İlk 10 Satırı")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Gözlem", df.shape[0])
    col2.metric("Toplam Değişken", df.shape[1])
    col3.metric("Sınıf Sayısı", df["Class"].nunique())

    st.write("### Sınıf Dağılımı")
    class_counts = df["Class"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    class_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Sınıf Dağılımı")
    st.pyplot(fig)

# ---------------------------------------------------
# TAB 2 – YAPISAL BİLGİLER
# ---------------------------------------------------
with tab2:
    st.subheader("Veri Seti Yapısı")

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Tipler")
        st.write(df.dtypes)

    with col2:
        st.write("### Özet İstatistikler")
        st.write(df.describe().T)

    st.write("### Eksik Değerler")
    st.write(df.isnull().sum())

# ---------------------------------------------------
# TAB 3 – DEĞİŞKEN DAĞILIMLARI
# ---------------------------------------------------
with tab3:
    st.subheader("Değişken Dağılım Analizi")

    col1, col2 = st.columns(2)

    # Histogram
    with col1:
        st.write(f"**Histogram – {feature_for_dist}**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[feature_for_dist], bins=20, edgecolor="black")
        ax.set_xlabel(feature_for_dist)
        ax.set_ylabel("Frekans")
        st.pyplot(fig)

    # Boxplot (sınıflara göre)
    with col2:
        st.write(f"**Boxplot – {feature_for_dist} (Class bazında)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="Class", y=feature_for_dist, ax=ax)
        ax.set_xlabel("Class")
        ax.set_ylabel(feature_for_dist)
        st.pyplot(fig)

# ---------------------------------------------------
# TAB 4 – KORELASYON
# ---------------------------------------------------
with tab4:
    st.subheader("Korelasyon Matrisi ve Isı Haritası")

    corr = df.drop(columns=["Class"]).corr()

    st.write("### Korelasyon Matrisi (Sayısal Değişkenler)")
    st.dataframe(corr)

    st.write("### Korelasyon Isı Haritası")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------------------------------------------------
# TAB 5 – PCA ANALİZİ
# ---------------------------------------------------
with tab5:
    st.subheader("PCA (Principal Component Analysis)")

    feature_cols = [
        "Alcohol",
        "Malic_acid",
        "Ash",
        "Alcalinity_of_ash",
        "Magnesium",
        "Total_phenols",
        "Flavanoids",
        "Nonflavanoid_phenols",
        "Proanthocyanins",
        "Color_intensity",
        "Hue",
        "OD280_OD315",
        "Proline",
    ]

    X = df[feature_cols].values
    y = df["Class"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    st.write("**Açıklanan Varyans Oranları:**", pca.explained_variance_ratio_)
    st.write(f"**Toplam Açıklanan Varyans:** {pca.explained_variance_ratio_.sum():.2%}")

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y,
        cmap="viridis",
        alpha=0.8,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA – İlk 2 Bileşen (Sınıflara Göre Renkli)")
    legend1 = ax.legend(
        *scatter.legend_elements(),
        title="Class",
        loc="best"
    )
    ax.add_artist(legend1)
    st.pyplot(fig)

# ---------------------------------------------------
# TAB 6 – RANDOM FOREST SINIFLANDIRMA
# ---------------------------------------------------
with tab6:
    st.subheader("Random Forest Sınıflandırma Modeli")

    feature_cols = [
        "Alcohol",
        "Malic_acid",
        "Ash",
        "Alcalinity_of_ash",
        "Magnesium",
        "Total_phenols",
        "Flavanoids",
        "Nonflavanoid_phenols",
        "Proanthocyanins",
        "Color_intensity",
        "Hue",
        "OD280_OD315",
        "Proline",
    ]

    X = df[feature_cols]
    y = df["Class"]

    # Ölçekleme (Random Forest için zorunlu değil ama tutarlı olsun diye)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    test_size = st.slider("Test oranı (validation size)", 0.1, 0.5, 0.2, 0.05)
    n_estimators = st.slider("Ağaç sayısı (n_estimators)", 50, 300, 100, 50)
    max_depth = st.slider("Maksimum derinlik (max_depth)", 2, 20, 8)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.2%}")
    col2.metric("Train size", len(X_train))
    col3.metric("Test size", len(X_test))

    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.write("### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).T)

    st.write("### Feature Importance")
    fi = pd.DataFrame(
        {"Feature": feature_cols, "Importance": rf.feature_importances_}
    ).sort_values("Importance", ascending=False)
    st.dataframe(fi, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=fi,
        x="Importance",
        y="Feature",
        ax=ax,
        orient="h",
    )
    ax.set_title("Feature Importance (Random Forest)")
    st.pyplot(fig)

# ---------------------------------------------------
# TAB 7 – DASHBOARD / ÖZET
# ---------------------------------------------------
with tab7:
    st.subheader("Dashboard ve Yönetsel Özet")

    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Gözlem", df.shape[0])
    col2.metric("Sınıf Sayısı", df["Class"].nunique())
    col3.metric("Ortalama Alkol", f"{df['Alcohol'].mean():.2f}")

    feature_cols = [
        "Alcohol", "Malic_acid", "Ash", "Alcalinity_of_ash", "Magnesium",
        "Total_phenols", "Flavanoids", "Nonflavanoid_phenols",
        "Proanthocyanins", "Color_intensity", "Hue", "OD280_OD315", "Proline",
    ]

    st.write("### Sınıfa Göre Ortalama Özellikler")
    st.dataframe(df.groupby("Class")[feature_cols].mean().round(2))

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Sınıfa Göre Ortalama Alkol**")
        fig, ax = plt.subplots(figsize=(6, 4))
        df.groupby("Class")["Alcohol"].mean().plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean Alcohol")
        st.pyplot(fig)

    with col2:
        st.write("**Sınıfa Göre Ortalama Proline**")
        fig, ax = plt.subplots(figsize=(6, 4))
        df.groupby("Class")["Proline"].mean().plot(kind="bar", ax=ax, color="orange")
        ax.set_ylabel("Mean Proline")
        st.pyplot(fig)
