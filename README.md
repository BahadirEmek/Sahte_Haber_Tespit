# Yapay Zeka Destekli Sahte Haber Tespit Sistemi

Bu proje, haber metinlerini analiz ederek girilen haberin **Fake** veya **Real** olarak sınıflandırılmasını amaçlayan makine öğrenmesi tabanlı bir web uygulamasıdır.

Projede doğal dil işleme yöntemleri kullanılarak haber metinleri ön işleme aşamasından geçirilmekte, ardından eğitilmiş sınıflandırma modeli ile tahmin yapılmaktadır. Kullanıcı, haber metnini doğrudan arayüz üzerinden girebilir veya `.txt` formatında bir metin dosyası yükleyebilir. Sistem girilen metni analiz ederek tahmini sonucu ve oranını kullanıcıya gösterir.

---

## İçindekiler

- [Proje Amacı](#proje-amacı)
- [Problem Tanımı](#problem-tanımı)
- [Kullanılan Veri Seti](#kullanılan-veri-seti)
- [Kullanılan Yöntemler](#kullanılan-yöntemler)
- [Metin Ön İşleme Aşamaları](#metin-ön-i̇şleme-aşamaları)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Model Yapısı](#model-yapısı)
- [Model Performans Metrikleri](#model-performans-metrikleri)
- [Proje Dosya Yapısı](#proje-dosya-yapısı)
- [Dosyaların Görevleri](#dosyaların-görevleri)
- [Kurulum](#kurulum)
- [Veri Setini Hazırlama](#veri-setini-hazırlama)
- [Model Eğitimi](#model-eğitimi)
- [Uygulamayı Çalıştırma](#uygulamayı-çalıştırma)
- [Uygulama Özellikleri](#uygulama-özellikleri)
- [Ekran Görüntüleri](#ekran-görüntüleri)
- [Önemli Notlar](#önemli-notlar)
- [Proje Durumu](#proje-durumu)
- [Geliştirici](#geliştirici)

---

## Proje Amacı

Projenin amacı, kullanıcı tarafından girilen veya yüklenen bir haber metnini analiz ederek bunun gerçek haber mi yoksa sahte haber mi olduğunu tahmin eden bir sistem geliştirmek. Sistemde doğal dil işleme yöntemlerinden yararlanılarak metin üzerinde ön işleme yapılacak, ardından makine öğrenmesi veya Transformer tabanlı bir model ile sınıflandırma gerçekleştirilecektir. Sonuç kısmında ise kullanıcıya haber kaynağı ile ilgili anlaşılır bir bilgi verilecektir.
---

## Problem Tanımı

Araştırmalarım sonucunda sahte haberler çoğu zaman başlık yapısı, kelime seçimi, abartılı ifade kullanımı ve içerik akışı bakımından belirli belirtiler taşıyabiliyor. Ancak bunları kişisel olarak ayırt etmenin uğraştırıcı bir süreç olması ve hata yapma oranı yüksek bir alandır. Bu nedenle proje kapsamında haber metinleri üzerinde otomatik analiz yaparak metindeki dilsel özellikleri öğrenen bir model kurulacak. Böylece sistem, yeni gelen bir haber metnini daha önce öğrendiği örüntülere göre değerlendirebilecek.

---

## Kullanılan Veri Seti

Projede **Fake and Real News Dataset** kullanılmıştır. Bu veri seti gerçek ve sahte haberlerden oluşan metin verilerini içermektedir.

Veri seti iki temel dosyadan oluşmaktadır:

- `Fake.csv`
- `True.csv`

Bu dosyalar proje klasöründeki `data/` klasörü içine eklenmelidir.

Beklenen veri seti yapısı:

```text
data/
├── Fake.csv
└── True.csv
```

Not: Veri seti dosyaları büyük boyutlu olabileceği için GitHub reposuna eklenmemiştir. Modeli yeniden eğitmek isteyen kullanıcıların veri setini indirip `data/` klasörü içine yerleştirmesi gerekir.

---

## Kullanılan Yöntemler

Projede metin sınıflandırma süreci aşağıdaki adımlarla gerçekleştirilmiştir:

1. Veri setinin okunması
2. Sahte ve gerçek haberlerin etiketlenmesi
3. Haber başlığı ve haber metninin birleştirilmesi
4. Metin ön işleme adımlarının uygulanması
5. Metinlerin TF-IDF yöntemi ile sayısal vektörlere dönüştürülmesi
6. Logistic Regression modeli ile sınıflandırma yapılması
7. Model performansının değerlendirilmesi
8. Eğitilen modelin kaydedilmesi
9. Streamlit arayüzü üzerinden kullanıcıya sunulması

---

## Kullanılan Teknolojiler

Projede kullanılan temel teknolojiler şunlardır:

- Python
- pandas
- NumPy
- scikit-learn
- NLTK
- Streamlit
- joblib
- matplotlib
- GitHub

---

## Model Yapısı

Projede temel model yapısı olarak aşağıdaki yaklaşım kullanılmıştır:

```text
TF-IDF Vectorizer + Logistic Regression
```

TF-IDF yöntemi, haber metinlerini sayısal vektörlere dönüştürmek için kullanılmıştır. Logistic Regression modeli ise bu vektörler üzerinden haberlerin sahte veya gerçek olarak sınıflandırılmasını sağlamaktadır.

---

## Model Performans Metrikleri

Model başarısı aşağıdaki metriklerle değerlendirilmiştir:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

Model eğitimi tamamlandıktan sonra performans değerleri terminalde görüntülenir. Ayrıca proje içerisinde model performans çıktıları `models/metrics.json` dosyasına, confusion matrix görseli ise `screenshots/confusion_matrix.png` dosyasına kaydedilebilir.

---

## Proje Dosya Yapısı

Projenin temel dosya yapısı aşağıdaki gibidir:

```text
Sahte_Haber_Tespit/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── models/
│   ├── fake_news_model.joblib
│   └── metrics.json
│
├── notebooks/
│
├── screenshots/
│   ├── home_screen.png
│   ├── prediction_fake.png
│   ├── prediction_real.png
│   ├── model_metrics.png
│   └── confusion_matrix.png
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── app.py
│
├── README.md
├── requirements.txt
├── .gitignore
└── Sahte_Haber_Tespit_Dokumani.docx
```

Not: `data/` klasöründeki veri seti dosyaları ve `models/` klasöründeki model dosyası eğitim sonrasında oluşturulabilir. Büyük dosyalar `.gitignore` içinde tutulabilir.

---

## Dosyaların Görevleri

### `src/preprocessing.py`

Metin temizleme ve ön işleme işlemlerini içerir. Haber metinleri model eğitiminden ve tahmin işleminden önce bu dosyadaki fonksiyonlar aracılığıyla düzenlenir.

Temel görevleri:

- Metni küçük harfe dönüştürmek
- URL ve HTML ifadelerini temizlemek
- Noktalama işaretlerini ve gereksiz karakterleri kaldırmak
- Stopword temizleme işlemi yapmak
- Model için daha düzenli metin çıktısı üretmek

---

### `src/train_model.py`

Model eğitim sürecini içerir. `Fake.csv` ve `True.csv` dosyalarını okuyarak veri setini hazırlar, metinleri ön işleme aşamasından geçirir, TF-IDF dönüşümü uygular ve Logistic Regression modelini eğitir.

Temel görevleri:

- Veri setini yüklemek
- Sahte ve gerçek haberleri etiketlemek
- Eğitim ve test verisi ayırmak
- Modeli eğitmek
- Performans metriklerini hesaplamak
- Eğitilmiş modeli kaydetmek

---

### `src/predict.py`

Eğitilmiş modeli kullanarak yeni haber metinleri için tahmin üretir. Kullanıcıdan gelen metin önce temizlenir, ardından kaydedilmiş TF-IDF vectorizer ve sınıflandırma modeli ile analiz edilir.

Temel görevleri:

- Eğitilmiş modeli yüklemek
- Kullanıcı metnini ön işleme aşamasından geçirmek
- Fake News / Real News tahmini üretmek
- Güven oranını hesaplamak
- Tahmini etkileyen kelimeleri göstermek

---

### `src/app.py`

Streamlit tabanlı web arayüzünü içerir. Kullanıcı bu arayüz üzerinden haber metni girebilir, `.txt` dosyası yükleyebilir, örnek haberlerden seçim yapabilir ve model tahmin sonucunu görüntüleyebilir.

Temel görevleri:

- Kullanıcı arayüzünü oluşturmak
- Haber metni girişi almak
- Dosya yükleme alanı sunmak
- Model tahmin sonucunu göstermek
- Güven oranı ve performans metriklerini kullanıcıya sunmak

---

## Kurulum

Projeyi çalıştırmak için önce repoyu bilgisayara klonlayın:

```bash
git clone https://github.com/BahadirEmek/Sahte_Haber_Tespit.git
```

Proje klasörüne girin:

```bash
cd Sahte_Haber_Tespit
```

Sanal ortam oluşturun:

```bash
python -m venv .venv
```

Windows CMD için sanal ortamı aktif edin:

```bash
.venv\Scripts\activate
```

PowerShell kullanılıyorsa aşağıdaki komut kullanılabilir:

```powershell
.\.venv\Scripts\Activate.ps1
```

Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

---

## Veri Setini Hazırlama

Kaggle üzerinden **Fake and Real News Dataset** indirilmelidir. İndirilen dosyalar arasından `Fake.csv` ve `True.csv` dosyaları alınarak `data/` klasörüne yerleştirilmelidir.

Beklenen yapı:

```text
data/
├── Fake.csv
└── True.csv
```

Veri seti dosyaları doğru konuma eklenmeden model eğitimi başlatılmamalıdır.

---

## Model Eğitimi

Veri seti `data/` klasörüne eklendikten sonra model eğitimi için aşağıdaki komut çalıştırılır:

```bash
python src\train_model.py
```

Eğitim tamamlandıktan sonra model dosyası `models/` klasörü içine kaydedilir:

```text
models/fake_news_model.joblib
```

Eğer performans metrikleri kaydediliyorsa aşağıdaki dosya da oluşur:

```text
models/metrics.json
```

Confusion matrix çıktısı ise aşağıdaki konuma kaydedilebilir:

```text
screenshots/confusion_matrix.png
```

---

## Uygulamayı Çalıştırma

Model eğitildikten sonra Streamlit arayüzünü başlatmak için aşağıdaki komut kullanılır:

```bash
streamlit run src\app.py
```

Uygulama tarayıcıda otomatik açılmazsa aşağıdaki adres üzerinden erişilebilir:

```text
http://localhost:8501
```

---

## Uygulama Özellikleri

Uygulamada bulunan temel özellikler:

- Haber metni girişi
- `.txt` dosyası yükleme
- Fake News / Real News tahmini
- Güven oranı gösterimi
- Hazır örnek haber metinleri ile test
- Tahmini etkileyen kelimeleri görüntüleme
- Temizlenmiş metni inceleme
- Model performans metriklerini görüntüleme
- Confusion matrix çıktısını görüntüleme
- Kullanıcı dostu Streamlit arayüzü

---

## Kullanım Akışı

Uygulama temel olarak şu şekilde kullanılmaktadır:

1. Kullanıcı haber metnini metin kutusuna girer veya `.txt` dosyası yükler.
2. Sistem metni ön işleme aşamasından geçirir.
3. Temizlenen metin TF-IDF yöntemi ile sayısal vektöre dönüştürülür.
4. Eğitilmiş Logistic Regression modeli tahmin yapar.
5. Kullanıcıya haberin **Fake News** veya **Real News** olduğu gösterilir.
6. Tahmin sonucu güven oranı ile birlikte sunulur.

---

## Ekran Görüntüleri

Uygulama ekran görüntüleri `screenshots/` klasörü içinde tutulmaktadır.

Örnek ekran görüntüleri:

```text
screenshots/home_screen.png
screenshots/prediction_fake.png
screenshots/prediction_real.png
screenshots/model_metrics.png
screenshots/confusion_matrix.png
```

Ekran görüntüleri proje arayüzünü, tahmin sonucunu ve model performans bölümlerini göstermek için kullanılmıştır.

---

## Önemli Notlar

Bu model, İngilizce haber veri setiyle eğitilmiştir. Bu nedenle İngilizce haber metinlerinde daha sağlıklı sonuç verir. Türkçe haber metinleriyle yapılan testlerde sonuçlar yalnızca demo amaçlı değerlendirilmelidir.

Model, haberin doğruluğunu internette araştırmaz. Sadece eğitim verisinden öğrendiği dilsel örüntülere göre tahmin üretir.

Bu nedenle sistemin çıktısı kesin doğruluk kontrolü olarak değil, makine öğrenmesi tabanlı bir sınıflandırma tahmini olarak değerlendirilmelidir.

---

## Proje Durumu

Proje kapsamında aşağıdaki işlemler tamamlanmıştır:

- GitHub reposu oluşturuldu.
- Proposal dokümanı hazırlandı.
- Proje klasör yapısı oluşturuldu.
- Veri seti proje yapısına uygun hale getirildi.
- Metin ön işleme akışı oluşturuldu.
- TF-IDF ve Logistic Regression tabanlı model eğitildi.
- Streamlit arayüzü geliştirildi.
- Sahte ve gerçek haber tahminleri test edildi.
- Model performans metrikleri kontrol edildi.
- Ekran görüntüleri alındı.
- Proje teslim dosya yapısı hazırlandı.

---

## Geliştirici

**Mustafa Bahadır Emek**  
Bilişim Sistemleri ve Teknolojileri

---

## GitHub Repo Linki

```text
https://github.com/BahadirEmek/Sahte_Haber_Tespit
```