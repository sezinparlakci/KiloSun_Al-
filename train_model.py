import json
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

generation = pd.read_csv("data/Plant_1_Generation_Data.csv")
weather = pd.read_csv("data/Plant_1_Weather_Sensor_Data.csv")

# Aldığım verileri datetime formatına dönüştürdüm. Kendi verinize göre şekillendirin.
generation["DATE_TIME"] = pd.to_datetime(generation["DATE_TIME"], dayfirst=True)
weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"])

generation = (
    generation.groupby("DATE_TIME")
    .agg(AC_POWER=("AC_POWER", "sum"), INVERTER_COUNT=("SOURCE_KEY", "nunique"))
    .reset_index()
)

# Eksik inverter kayıtları toplam üretimi düşürebilir sadece tüm inverterlerin zamnalarını tutuyorum.
generation = generation[generation["INVERTER_COUNT"] == generation["INVERTER_COUNT"].max()]

data = pd.merge(generation, weather, on="DATE_TIME")
print("Birleştirilmiş veri:", data.shape)

features = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
data = data[["DATE_TIME"] + features + ["AC_POWER"]]
print("Eksik değerler:\n", data.isnull().sum())
data = data.dropna().sort_values("DATE_TIME").reset_index(drop=True)

# Train / test ayrımı (ilk %80 eğitim, son %20 test)
split_index = int(len(data) * 0.8)
train = data.iloc[:split_index]
test = data.iloc[split_index:].copy()
x_train, y_train = train[features], train["AC_POWER"]
x_test, y_test = test[features], test["AC_POWER"]
print("Train:", train["DATE_TIME"].min(), "->", train["DATE_TIME"].max())
print("Test :", test["DATE_TIME"].min(), "->", test["DATE_TIME"].max())

# Gece ışınım sıfır olduğu zamanlarda model R^2 çok yüksek olabilir engellemek amaçlı gündüz verileri izole ediyorum.
day = (x_test["IRRADIATION"] > 0).to_numpy()

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    pred = np.clip(pred, 0, None)  # Negatif değerleri 0 a sabitliyor.
    predictions[name] = pred

    result = {
        "model": name,
        "mae": mean_absolute_error(y_test, pred),
        "r2": r2_score(y_test, pred),
        "mae_day": mean_absolute_error(y_test[day], pred[day]),
        "r2_day": r2_score(y_test[day], pred[day]),
    }
    results.append(result)
    print(f"\n{name}")
    print(f"MAE: {result['mae']:.2f} | R²: {result['r2']:.4f}")
    print(f"MAE (gündüz): {result['mae_day']:.2f} | R² (gündüz): {result['r2_day']:.4f}")

# En iyi model için en düşük mutlak hata( MAE) sahip olan modeli seçtim
best = min(results, key=lambda r: r["mae"])
best_name = best["model"]
best_model = models[best_name]
best_pred = predictions[best_name]
print("\nSeçilen model:", best_name)

# Gündüz hata standart sapmasının 2 katını sapma eşiği olarak tanımlıyorum
errors = y_test.to_numpy() - best_pred
threshold = 2 * errors[day].std()
print("Sapma eşiği:", round(threshold, 2))

# Modeli kaydetme aşaması 
os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/best_model.pkl")

info = {
    "best_model": best_name,
    "features": features,
    "mae": best["mae"],
    "r2": best["r2"],
    "mae_day": best["mae_day"],
    "r2_day": best["r2_day"],
    "deviation_threshold": float(threshold),
    "test_start": str(test["DATE_TIME"].min()),
    "test_end": str(test["DATE_TIME"].max()),
    "all_results": results,
}
with open("model/model_info.json", "w", encoding="utf-8") as f:
    json.dump(info, f, indent=4, ensure_ascii=False)

# Streamlit grafikleri için test tahminleri
test["PREDICTED_AC_POWER"] = best_pred
test["DEVIATION"] = test["AC_POWER"] - test["PREDICTED_AC_POWER"]
test.to_csv("model/test_predictions.csv", index=False)

print("\nModel ve sonuç dosyaları 'model' klasörüne kaydedildi.")