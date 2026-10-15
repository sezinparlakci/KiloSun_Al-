KiloSun Al+

Güneş enerjisi santrallerinin AC güç üretimini çevresel parametrelere göre tahmin eden ve üretim sapmalarını analiz eden Streamlit tabanlı yapay zeka uygulamasıdır.

Özellikler

- AC Power Tahmini: Ortam sıcaklığı, modül sıcaklığı ve ışınım verileriyle üretim tahmini.
- Üretim Sapması Analizi: Gerçek üretim girildiğinde model beklentisiyle kıyaslama ve sapma durumu tespiti.
- Görselleştirme: Gerçek/tahmin zaman serisi grafikleri ve parametre ilişkileri.

Teknolojiler

Python, pandas, NumPy, scikit-learn, matplotlib, Streamlit, joblib

cd KiloSun-Al
pip install -r requirements.txt
streamlit run App.py

Not: Eğitilmiş model (model/best_model.pkl) ve sonuç dosyaları repoda hazır olarak yer almaktadır. Bu nedenle uygulamayı çalıştırmak için yeniden model eğitimi yapmanıza gerek yoktur. Modeli sıfırdan eğitmek isterseniz veri setini data/ klasörüne ekleyip python train_model.py komutunu çalıştırabilirsiniz.
