# Yapay Zeka Destekli Sahte Haber Tespit Sistemi

Bu projede haber metinlerini analiz ederek bir haberin **Fake News** veya **Real News** olarak sınıflandırılmasını sağlayan bir makine öğrenmesi uygulaması geliştirdim.

Projede doğal dil işleme ve makine öğrenmesi yöntemlerini kullandım. Kullanıcıdan alınan haber metni önce temizleniyor, ardından TF-IDF yöntemiyle sayısal hale getiriliyor ve eğitilmiş Logistic Regression modeli ile tahmin yapılıyor. Sonuçlar Streamlit ile hazırladığım web arayüzü üzerinden gösteriliyor.

Bu sistem haberin doğruluğunu internetten kontrol etmiyor. Model, eğitim verisinde gördüğü kelime kullanımları ve metin örüntülerine göre tahmin üretiyor. Bu yüzden sonuçları kesin doğruluk kontrolü olarak değil, makine öğrenmesi tabanlı bir sınıflandırma sonucu olarak değerlendirmek gerekir.

---

## Projenin Amacı

Bu projedeki amacım, sahte ve gerçek haberleri metin üzerinden ayırt etmeye çalışan basit ama çalışan bir yapay zeka sistemi oluşturmaktı.

Günümüzde sosyal medya ve internet üzerinden çok fazla haber içeriği paylaşılıyor. Bu haberlerin bir kısmı doğru kaynaklara dayanırken, bazıları eksik, abartılı veya tamamen yanlış bilgiler içerebiliyor. Özellikle doğrulanmamış haberlerin hızlı yayılması, kullanıcıların yanlış bilgiye maruz kalmasına neden olabiliyor.

Bu problemden yola çıkarak, haber metinlerini analiz eden ve bunları **Fake News** veya **Real News** olarak sınıflandıran bir makine öğrenmesi sistemi geliştirdim.

---

## Kullanılan Veri Seti

Projede **Fake and Real News Dataset** veri setini kullandım.

Veri seti iki ana dosyadan oluşuyor:

```
data/
├── Fake.csv
└── True.csv
```

`Fake.csv` içindeki haberler sahte haber, `True.csv` içindeki haberler gerçek haber olarak etiketlendi.

Etiketleme yapısı:

```
0 → Fake News
1 → Real News
```

Veri seti dosyaları büyük olduğu için GitHub reposuna eklenmedi. Projeyi yeniden çalıştırmak isteyen kişinin bu dosyaları indirip `data/` klasörüne koyması gerekir.

---

## Kullanılan Teknolojiler

| Teknoloji | Sürüm | Kullanım Amacı |
|---|---|---|
| Python | 3.10+ | Projenin ana programlama dili |
| pandas | — | Veri setini okuma ve düzenleme |
| NumPy | — | Sayısal matris işlemleri |
| scikit-learn | — | TF-IDF vektörizasyonu, Logistic Regression modeli ve metrik hesaplama |
| NLTK | — | İngilizce stopword listesi ile metin ön işleme |
| joblib | — | Eğitilen model ve vektörizerin diske kaydedilmesi ve yüklenmesi |
| matplotlib | — | Confusion Matrix grafiğinin oluşturulması ve kaydedilmesi |
| Streamlit | — | Kullanıcı arayüzü (web tabanlı) |
| GitHub | — | Proje sürüm kontrolü ve teslim |

---

## Sistem Mimarisi

```
Kullanıcı Metni
      │
      ▼
┌─────────────────┐
│  Ön İşleme      │  preprocessing.py
│  (temizleme,    │  • küçük harf
│   stopword)     │  • URL / HTML / noktalama temizleme
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TF-IDF         │  scikit-learn TfidfVectorizer
│  Vektörizasyon  │  max_features=10000, ngram=(1,2)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Logistic       │  scikit-learn LogisticRegression
│  Regression     │  max_iter=1000
└────────┬────────┘
         │
         ▼
  Fake News / Real News + Güven Skoru
```

---

## Metin Ön İşleme

Modelin daha doğru çalışması için haber metinleri eğitim öncesinde temizlendi.

Uygulanan işlemler:

1. Küçük harfe çevirme
2. URL temizleme (`https://...`, `www....`)
3. HTML etiketlerini kaldırma (`<b>`, `<p>` vb.)
4. Noktalama işaretlerini kaldırma
5. Sayısal ifadeleri kaldırma
6. Fazla boşlukları düzenleme
7. İngilizce stopword temizleme (NLTK, fallback liste)
8. Boş metin kontrolü

Bu işlemler `src/preprocessing.py` dosyasında `TextPreprocessor` sınıfı içinde yer alıyor.

---

## Model Eğitimi ve Değerlendirme

Eğitim süreci şu adımlardan oluşuyor:

1. `Fake.csv` ve `True.csv` dosyaları okunarak birleştirildi
2. Haber başlığı (`title`) ve haber metni (`text`) birleştirilerek `content` sütunu oluşturuldu
3. Metinler `TextPreprocessor` ile temizlendi
4. Veri seti %80 eğitim, %20 test olarak ayrıldı (stratified split)
5. `TfidfVectorizer` eğitim verisi üzerinde fit edildi
6. `LogisticRegression` modeli eğitildi
7. Model performansı test seti üzerinde değerlendirildi
8. Model ve vektörizer `models/fake_news_model.joblib` olarak kaydedildi

Değerlendirme metrikleri:

- **Accuracy** — genel doğruluk oranı
- **Precision** — sahte haber olarak işaretlenenlerin ne kadarı gerçekten sahteydi
- **Recall** — gerçek sahte haberlerin ne kadarı yakalandı
- **F1-Score** — precision ve recall'ın harmonik ortalaması
- **Confusion Matrix** — sınıf bazında doğru/yanlış tahmin dağılımı

---

## Tahmin Aşaması

Kullanıcıdan alınan metin şu adımlardan geçiyor:

1. `TextPreprocessor.clean_text()` ile temizleniyor
2. Kaydedilmiş `TfidfVectorizer` ile vektörize ediliyor
3. `LogisticRegression.predict()` ile tahmin yapılıyor
4. `predict_proba()` ile güven skoru hesaplanıyor
5. TF-IDF ağırlıkları ve model katsayıları kullanılarak en etkili 10 kelime çıkarılıyor

---

## Dosyaların Görevleri

### `src/preprocessing.py`

Metin temizleme işlemleri. Hem model eğitimi sırasında hem de kullanıcıdan gelen yeni metinler için bu dosyadaki `TextPreprocessor` sınıfı kullanılıyor.

### `src/train_model.py`

Model eğitim süreci. `Fake.csv` ve `True.csv` dosyalarını okuyup veriyi hazırlar, TF-IDF dönüşümü yapar, Logistic Regression modelini eğitir, metrikleri `models/metrics.json` olarak ve Confusion Matrix'i `screenshots/confusion_matrix.png` olarak kaydeder.

Çalıştırmak için:
```bash
python src/train_model.py
```

### `src/predict.py`

Eğitilmiş modeli kullanarak yeni haber metinleri için tahmin yapıyor. Metni temizliyor, modele gönderiyor, güven skoru ve önemli kelimeleri hesaplayıp döndürüyor.

### `src/app.py`

Streamlit web arayüzü. İki sekme içeriyor:
- **Metin Analizi** — haber metni girişi, örnek haberler, `.txt` dosyası yükleme ve tahmin sonucu
- **Model Performansı** — accuracy/precision/recall/F1 metrikleri ve Confusion Matrix

Çalıştırmak için:
```bash
streamlit run src/app.py
```

---

## Proje Klasör Yapısı

```
Sahte_Haber_Tespit/
├── data/
│   ├── Fake.csv          ← veri seti (repoya eklenmedi)
│   └── True.csv          ← veri seti (repoya eklenmedi)
├── examples/
│   ├── fake_1.txt        ← örnek sahte haber metni
│   ├── fake_2.txt
│   ├── fake_3.txt
│   ├── real_1.txt        ← örnek gerçek haber metni
│   ├── real_2.txt
│   └── real_3.txt
├── models/
│   ├── fake_news_model.joblib   ← eğitilmiş model
│   └── metrics.json             ← performans metrikleri
├── screenshots/
│   └── confusion_matrix.png
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── app.py
├── requirements.txt
└── README.md
```

---

## Karşılaştığım Durumlar

Projeyi geliştirirken fark ettiğim en önemli nokta, modelin İngilizce veri setiyle eğitildiği için Türkçe haberlerde güvenilir sonuç vermemesiydi.

İlk testlerde Türkçe doğru haber örneklerinin bazen sahte haber olarak sınıflandırıldığını gördüm. Bunun sebebi modelin Türkçe metinlerle hiç eğitilmemiş olmasıdır; Türkçe kelimeler model için anlamsız gürültüye dönüşüyor. Bu nedenle güvenilir testleri İngilizce haber örnekleri üzerinden yaptım ve arayüze bu konuda kullanıcıyı bilgilendiren bir uyarı ekledim.

Bu durum bana veri seti dili ile test edilen metnin dilinin uyumlu olmasının ne kadar kritik olduğunu gösterdi.

---

## Önemli Not

Bu model İngilizce haber veri setiyle eğitilmiştir. İngilizce haber metinlerinde güvenilir sonuç verir.

Türkçe haber metinleriyle yapılan testler yalnızca demo amaçlı değerlendirilmelidir.

Model haberin doğruluğunu internette araştırmaz. Sadece eğitim verisinden öğrendiği kelime kullanımları ve metin örüntülerine göre sınıflandırma yapar.

---

## Geliştirilebilecek Yönler

- Türkçe haber veri seti ile yeniden eğitim yapılabilir
- BERT veya benzeri transformer tabanlı dil modelleri denenebilir
- Farklı makine öğrenmesi modelleri (Random Forest, SVM, XGBoost) karşılaştırılabilir
- Daha büyük ve dengeli veri setleri kullanılabilir
- Haber kaynağı ve yayın tarihi gibi meta veriler modele dahil edilebilir
- Girilen haberin internetteki kaynaklarla gerçek zamanlı karşılaştırılması sağlanabilir

---

## Proje Durumu

Tamamlanan aşamalar:

- [x] GitHub reposu oluşturuldu
- [x] Proje klasör yapısı oluşturuldu
- [x] Veri seti projeye uygun şekilde kullanıldı
- [x] Metin ön işleme adımları hazırlandı
- [x] TF-IDF ve Logistic Regression ile model eğitimi yapıldı
- [x] Model performansı metriklerle değerlendirildi
- [x] Streamlit arayüzü oluşturuldu ve sekme tabanlı yapıya geçildi
- [x] Örnek haber metinleri eklendi (İngilizce fake/real)
- [x] Ekran görüntüleri alındı
- [x] Teslim için gerekli dosyalar düzenlendi

---

## Demo Video

[Demo videoyu izlemek için tıklayın](https://github.com/BahadirEmek/Sahte_Haber_Tespit/releases/tag/1.0)

---

## Geliştirici

**Mustafa Bahadır Emek**  
Bilişim Sistemleri ve Teknolojileri

GitHub:
```
https://github.com/BahadirEmek/Sahte_Haber_Tespit
```
