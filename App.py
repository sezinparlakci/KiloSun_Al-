import json
import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="KiloSun Al+")
st.image("header.jpg", width=650)

# Önce model eğitim dosyalarım var mı onu kontrol ediyorum
for file in ["best_model.pkl", "model_info.json", "test_predictions.csv"]:
    if not os.path.exists("model/" + file):
        st.error("model/" + file + " bulunamadı. Önce python train_model.py çalıştırın.")
        st.stop()

# Eğitilen model ve metrik bilgileri bellek optimizasyonu için önbelleğe alınarak yükleniyor.
@st.cache_resource
def load_model():
    return joblib.load("model/best_model.pkl")

model = load_model()
with open("model/model_info.json", encoding="utf-8") as f:
    info = json.load(f)
preds = pd.read_csv("model/test_predictions.csv", parse_dates=["DATE_TIME"])
threshold = info["deviation_threshold"]

# sayıyı türkçe formatta yazmak için 16000 -> 16.000 gibi
def tr(value, fmt=",.0f"):
    return format(value, fmt).replace(",", ".")


st.title("KiloSun Al+")
st.markdown(
    """
    ### Fotovoltaik Güneş Enerjisi Üretim ve Sapma Analitiği
    
    Bu sistem, güneş enerjisi santrallerinin anlık AC güç üretimini çevresel parametreler 
    aracılığıyla tahmin eden ve üretim sapmalarını tespit eden yapay zeka tabanlı bir analiz platformudur.
    
    * **Veri Kaynağı:** Kaggle Solar Power Generation Data (Plant 1)
    * **Kullanılan Parametreler:** Ortam Sıcaklığı, Modül Sıcaklığı ve Güneş Işınımı
    """
)
st.subheader("Model performansı")
st.write(f"Kullanılan model: **{info['best_model']}**")

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{info['mae']:.0f} kW")
c2.metric("R²", f"{info['r2']:.4f}")
c3.metric("MAE (gündüz)", f"{info['mae_day']:.0f} kW")
c4.metric("R² (gündüz)", f"{info['r2_day']:.4f}")


st.subheader("AC Power tahmini")

col1, col2, col3 = st.columns(3)
ambient = col1.number_input("Ortam sıcaklığı (°C)", -10.0, 60.0, 28.0, 0.5)
module = col2.number_input("Modül sıcaklığı (°C)", -10.0, 100.0, 40.0, 0.5)
irradiation = col3.number_input("Güneş ışınımı (kW/m²)", 0.0, 1.5, 0.6, 0.01)

actual = st.number_input("Gerçek AC Power (kW), isteğe bağlı", 0, 100000, 0, 100)

if st.button("Predict"):
    x_new = pd.DataFrame([[ambient, module, irradiation]], columns=info["features"])
    pred = max(model.predict(x_new)[0], 0)
    st.metric("Tahmini AC Power", f"{tr(pred)} kW")

    # Gerçek üretim girildiyse tahminle karşılaştırıyorum
    if actual > 0 and irradiation > 0:
        diff = actual - pred
        st.write(f"Gerçek: {tr(actual)} kW | Tahmin: {tr(pred)} kW | Fark: {tr(diff, '+,.0f')} kW")
        if diff < -threshold:
            st.warning("Beklenenden düşük üretim")
        elif diff > threshold:
            st.info("Beklenenden yüksek üretim")
        else:
            st.success("Üretim beklenen aralıkta")
        st.caption("Bu kesin bir arıza tespiti değildir, sadece üretimdeki sapmayı gösterir.")


st.subheader("Grafikler")

# Gerçek ve tahmin edilen üretim
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(preds["DATE_TIME"], preds["AC_POWER"], label="Gerçek")
ax.plot(preds["DATE_TIME"], preds["PREDICTED_AC_POWER"], label="Tahmin")
ax.set_xlabel("Zaman")
ax.set_ylabel("AC Power (kW)")
ax.set_title("Gerçek ve tahmin edilen AC Power")
ax.legend()
fig.autofmt_xdate()
st.pyplot(fig)

# Işınım ve modül sıcaklığı ile üretim ilişkisi
left, right = st.columns(2)

fig, ax = plt.subplots()
ax.scatter(preds["IRRADIATION"], preds["AC_POWER"], s=8)
ax.set_xlabel("Güneş ışınımı (kW/m²)")
ax.set_ylabel("AC Power (kW)")
ax.set_title("Işınım - AC Power")
left.pyplot(fig)

fig, ax = plt.subplots()
ax.scatter(preds["MODULE_TEMPERATURE"], preds["AC_POWER"], s=8, color="orange")
ax.set_xlabel("Modül sıcaklığı (°C)")
ax.set_ylabel("AC Power (kW)")
ax.set_title("Modül sıcaklığı - AC Power")
right.pyplot(fig)

# Sadece gündüz için üretim sapması
day = preds[preds["IRRADIATION"] > 0]
low = day[day["DEVIATION"] < -threshold]

st.subheader("Üretim sapması")
st.write(
    f"Gündüz {len(day)} ölçümün {len(low)} tanesinde üretim beklenenden düşük "
    f"(eşik: ±{tr(threshold)} kW)."
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(day["DATE_TIME"], day["DEVIATION"], ".", markersize=4)
ax.axhline(threshold, color="red", linestyle="--")
ax.axhline(-threshold, color="red", linestyle="--")
ax.set_xlabel("Zaman")
ax.set_ylabel("Gerçek - Tahmin (kW)")
ax.set_title("Gündüz üretim sapması")
fig.autofmt_xdate()
st.pyplot(fig)