# 🍷 Wine Dataset Analysis Dashboard

## Proje Açıklaması

Bu proje, **UCI Wine veri seti** kullanarak Streamlit ile geliştirilmiş kapsamlı bir veri analizi ve makine öğrenmesi uygulamasıdır. Veri Bilimi için Programlama dersi kapsamında oluşturulan bu dashboard, veri keşfi, istatistiksel analiz ve sınıflandırma modellemesi işlemlerini içermektedir.

## 📋 Özellikler

Uygulama **7 ana sekme** üzerinden aşağıdaki analizleri sunar:

### 1. 📋 Genel Bakış (Overview)
- Veri setinin ilk 10 satırının görüntülenmesi
- Toplam gözlem ve değişken sayısı
- Sınıf dağılımı grafiği

### 2. 🔍 Yapısal Bilgiler (Structure)
- Değişken tipleri
- Özet istatistikler
- Eksik değer analizi

### 3. 📈 Değişken Dağılımları (Distributions)
- Seçili değişkene ait histogram
- Sınıflara göre boxplot analizi
- Etkileşimli değişken seçimi

### 4. 📊 Korelasyon Analizi (Correlation)
- Korelasyon matrisi
- Isı haritası (heatmap) görselleştirmesi

### 5. 🧠 PCA Analizi (Principal Component Analysis)
- 13 boyutlu özellik uzayının 2 boyuta indirgenmesi
- Açıklanan varyans oranları
- Sınıflara göre renklendirilmiş PCA grafiği

### 6. 🤖 Random Forest Sınıflandırması (Classification)
- Model eğitim ve test parametreleri (interactive sliders)
- Karışıklık matrisi (confusion matrix)
- Sınıflandırma raporu
- Feature importance analizi

### 7. 📍 Dashboard & Özet (Summary)
- Sınıflara göre ortalama özellikler
- Çeşitli görsel özetler
- Yönetsel öneriler

## 🛠️ Gerekli Kütüphaneler

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## 🚀 Kullanım

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

2. Uygulamayı çalıştırın:
```bash
streamlit run app.py
```

3. Tarayıcınızda açılan uygulamada farklı sekmeleri keşfedin ve analiz parametrelerini ayarlayın.

## 📁 Dosya Yapısı

```
├── app.py           # Ana Streamlit uygulaması
├── wine.data        # UCI Wine veri seti
├── wine.names       # Veri seti açıklaması
└── README.md        # Proje dokümantasyonu
```

## 📊 Veri Seti Bilgileri

- **Kaynak:** UCI Machine Learning Repository - Wine Dataset
- **Örnek Sayısı:** 178 gözlem
- **Özellikleri:** 13 kimyasal özellik
- **Sınıflar:** 3 farklı şarap tipi

## 💡 Temel Bulgular

- Veri seti üç sınıf şarap içermektedir
- PCA analizi ile yüksek boyutsal verileri 2D'ye projekte edilebilmektedir
- Random Forest modeli yüksek doğruluk oranları elde etmektedir
- Feature importance analizi, hangi kimyasal özelliklerin sınıflandırmada önemli olduğunu göstermektedir

## 👤 Yazar

Rana Özcan

---

