import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

st.title("📊 Uji Regresi Linear Sederhana & Asumsi Klasik")

uploaded_file = st.file_uploader("📁 Upload file Excel (.xlsx)", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ Data berhasil diunggah!")
    st.write("📄 Data preview:")
    st.dataframe(df)

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    x_col = st.selectbox("📌 Pilih variabel X (Prediktor):", options=numeric_columns)
    y_col = st.selectbox("🎯 Pilih variabel Y (Respon):", options=numeric_columns)

    if x_col and y_col:
        X = df[[x_col]]
        X = sm.add_constant(X)
        y = df[y_col]

        model = sm.OLS(y, X).fit()
        st.subheader("📈 Hasil Regresi Linear")
        st.text(model.summary())

        residuals = model.resid

        # Uji Normalitas
        st.subheader("🧪 Uji Normalitas (Shapiro-Wilk)")
        stat, p = shapiro(residuals)
        st.write(f"Statistik: {stat:.4f}, p-value: {p:.4f}")
        st.markdown("✅ Residual normal" if p > 0.05 else "❌ Residual tidak normal")

        # Histogram Residual
        fig, ax = plt.subplots()
        ax.hist(residuals, bins=10, edgecolor='black')
        ax.set_title("Histogram Residual")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)

        # Multikolinearitas (VIF)
        st.subheader("🔍 Uji Multikolinearitas (VIF)")
        vif_df = pd.DataFrame()
        vif_df["Feature"] = X.columns
        vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        st.dataframe(vif_df)

        # Heteroskedastisitas
        st.subheader("📉 Uji Heteroskedastisitas (Breusch-Pagan)")
        bp = het_breuschpagan(residuals, X)
        labels = ['LM Statistic', 'LM p-value', 'F-statistic', 'F p-value']
        result = dict(zip(labels, bp))
        st.write(result)
        st.markdown("✅ Tidak ada heteroskedastisitas" if bp[3] > 0.05 else "❌ Terdapat heteroskedastisitas")

        # Autokorelasi
        st.subheader("🔁 Uji Autokorelasi (Durbin-Watson)")
        dw = durbin_watson(residuals)
        st.write(f"Durbin-Watson: {dw:.4f}")
        st.markdown("✅ Tidak ada autokorelasi" if 1.5 < dw < 2.5 else "❌ Terdapat autokorelasi")
