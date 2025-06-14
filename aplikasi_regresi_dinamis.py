import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

st.set_page_config(page_title="Uji Regresi Linear", layout="centered")
st.title("📊 Uji Regresi Linear Sederhana & Asumsi Klasik")

uploaded_file = st.file_uploader("📁 Upload file Excel (.xlsx)", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ Data berhasil diunggah!")
    st.write("📄 Data Preview:")
    st.dataframe(df)

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    x_col = st.selectbox("📌 Pilih variabel X (Prediktor):", options=numeric_columns)
    y_col = st.selectbox("🎯 Pilih variabel Y (Respon):", options=numeric_columns)

    if x_col and y_col:
        if x_col == y_col:
            st.warning("❗ Variabel X dan Y tidak boleh sama.")
        else:
            X = df[[x_col]]
            X = sm.add_constant(X)
            y = df[y_col]

            model = sm.OLS(y, X).fit()
            residuals = model.resid

            # Tampilkan hasil regresi
            st.subheader("📈 Hasil Regresi Linear")
            html_summary = model.summary().as_html()
            st.markdown(html_summary, unsafe_allow_html=True)

            # Interpretasi singkat
            st.markdown("### 📌 Interpretasi Singkat")
            st.markdown(f"- **Intercept (const):** {model.params[0]:.4f}")
            st.markdown(f"- **Koefisien {x_col}:** {model.params[1]:.4f}")
            st.markdown(f"- **R-squared:** {model.rsquared:.4f} ➡️ {model.rsquared*100:.2f}% variasi pada `{y_col}` dijelaskan oleh `{x_col}`")

            # Uji F dan Uji t
            st.subheader("📉 Uji F dan Uji t")

            # Uji F
            f_stat = float(model.fvalue)
            f_pval = float(model.f_pvalue)
            st.markdown(f"**Uji F (model keseluruhan):**")
            st.write(f"F-statistic: {f_stat:.4f}")
            st.write(f"p-value: {f_pval:.4g}")
            st.markdown("✅ Model signifikan secara statistik" if f_pval < 0.05 else "❌ Model tidak signifikan secara statistik")

            # Uji t (koefisien)
            st.markdown("**Uji t (koefisien regresi):**")
            for var in model.params.index:
                t_val = model.tvalues[var]
                p_val = model.pvalues[var]
                st.write(f"- **{var}**: t = {t_val:.4f}, p-value = {p_val:.4g} → "
                        + ("✅ Signifikan" if p_val < 0.05 else "❌ Tidak signifikan"))

            # Plot Regresi
            st.subheader("📈 Plot Regresi Linear")
            fig2, ax2 = plt.subplots()
            ax2.scatter(df[x_col], y, label='Data Asli', color='skyblue')
            ax2.plot(df[x_col], model.predict(X), color='red', label='Garis Regresi')
            ax2.set_xlabel(x_col)
            ax2.set_ylabel(y_col)
            ax2.set_title("Plot Regresi Linear")
            ax2.legend()
            st.pyplot(fig2)

            # Uji Normalitas
            st.subheader("📉 Uji Normalitas (Shapiro-Wilk)")
            stat, p = shapiro(residuals)
            st.write(f"Statistik: {stat:.4f}, p-value: {p:.4f}")
            st.markdown("✅ Residual normal" if p > 0.05 else "❌ Residual tidak normal")

            fig, ax = plt.subplots()
            ax.hist(residuals, bins=10, edgecolor='black')
            ax.set_title("Histogram Residual")
            ax.set_xlabel("Residual")
            ax.set_ylabel("Frekuensi")
            st.pyplot(fig)

            # Uji Multikolinearitas (VIF)
            st.subheader("📈 Uji Multikolinearitas (VIF)")
            vif_df = pd.DataFrame()
            vif_df["Feature"] = X.columns
            vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            st.dataframe(vif_df)
            st.markdown("""
            - VIF > 10: ❌ Menunjukkan multikolinearitas tinggi
            - VIF < 5: ✅ Multikolinearitas rendah
            """)

            # Uji Heteroskedastisitas
            st.subheader("📉 Uji Heteroskedastisitas (Breusch-Pagan)")
            bp = het_breuschpagan(residuals, X)
            labels = ['LM Statistic', 'LM p-value', 'F-statistic', 'F p-value']
            result = dict(zip(labels, bp))
            st.write(result)
            st.markdown("✅ Tidak ada heteroskedastisitas" if bp[3] > 0.05 else "❌ Terdapat heteroskedastisitas")

            # Uji Autokorelasi
            st.subheader("📈 Uji Autokorelasi (Durbin-Watson)")
            dw = durbin_watson(residuals)
            st.write(f"Durbin-Watson: {dw:.4f}")
            st.markdown("✅ Tidak ada autokorelasi" if 1.5 < dw < 2.5 else "❌ Terdapat autokorelasi")

            st.markdown("---")
            st.info("Ingin menganalisis ulang? Silakan upload file baru atau ubah pilihan variabel.")
