# Yapay Zeka Destekli Sahte Haber Tespit Sistemi

Bu projede haber metinlerini analiz ederek bir haberin **Fake News** veya **Real News** olarak sınıflandırılmasını sağlayan bir makine öğrenmesi uygulaması geliştirdim.

Projede temel olarak doğal dil işleme ve makine öğrenmesi yöntemlerini kullandım. Kullanıcıdan alınan haber metni önce temizleniyor, daha sonra TF-IDF yöntemi ile sayısal hale getiriliyor ve eğitilmiş Logistic Regression modeli ile tahmin yapılıyor. Sonuçlar Streamlit ile hazırladığım web arayüzü üzerinden gösteriliyor.

Bu sistem haberin doğruluğunu internetten kontrol etmiyor. Model, eğitim verisinde gördüğü kelime kullanımları ve metin örüntülerine göre tahmin üretiyor. Bu yüzden sonuçları kesin doğruluk kontrolü olarak değil, makine öğrenmesi tabanlı bir sınıflandırma sonucu olarak değerlendirmek gerekir.

---

## Projenin Amacı

Bu projedeki amacım, sahte ve gerçek haberleri metin üzerinden ayırt etmeye çalışan basit ama çalışan bir yapay zeka sistemi oluşturmaktı.

Günümüzde sosyal medya ve internet üzerinden çok fazla haber içeriği paylaşılıyor. Bu haberlerin bir kısmı doğru kaynaklara dayanırken, bazıları eksik, abartılı veya tamamen yanlış bilgiler içerebiliyor. Özellikle doğrulanmamış haberlerin hızlı yayılması, kullanıcıların yanlış bilgiye maruz kalmasına neden olabiliyor.

Bu problemden yola çıkarak, haber metinlerini analiz eden ve bunları **Fake News** veya **Real News** olarak sınıflandıran bir sistem geliştirdim.

---

## Kullanılan Veri Seti

Projede **Fake and Real News Dataset** veri setini kullandım.

Veri seti iki ana dosyadan oluşuyor:

```text
Fake.csv
True.csv
```

Ben bu dosyalardan `Fake.csv` içindeki haberleri sahte haber, `True.csv` içindeki haberleri ise gerçek haber olarak kullandım.

Etiketleme yapısı şu şekilde:

```text
0 → Fake News
1 → Real News
```

Veri seti dosyalarının proje içinde şu klasörde olması gerekiyor:

```text
data/
├── Fake.csv
└── True.csv
```

Veri seti dosyaları büyük olabildiği için GitHub reposuna eklenmeyebilir. Projeyi yeniden çalıştırmak isteyen kişinin bu dosyaları indirip `data/` klasörüne koyması gerekir.

---

## Kullandığım Yöntem

Projede haber metinlerini sınıflandırmak için şu adımları izledim:

1. Fake ve gerçek haber verilerini okudum.
2. Haberleri `Fake News` ve `Real News` olarak etiketledim.
3. Haber başlığı ve haber metnini birleştirdim.
4. Metinleri ön işleme aşamasından geçirdim.
5. TF-IDF yöntemi ile metinleri sayısal hale getirdim.
6. Logistic Regression modeli ile eğitim yaptım.
7. Modelin başarısını accuracy, precision, recall ve F1-score metrikleri ile kontrol ettim.
8. Eğitilen modeli kaydettim.
9. Streamlit arayüzü ile kullanıcıdan haber metni alıp tahmin sonucunu gösterdim.

---

## Metin Ön İşleme

Modelin daha düzgün çalışması için haber metinlerini eğitimden önce temizledim.

Uyguladığım temel işlemler:

- Küçük harfe çevirme
- URL temizleme
- HTML etiketlerini temizleme
- Noktalama işaretlerini kaldırma
- Sayısal ifadeleri temizleme
- Fazla boşlukları düzenleme
- Stopword temizleme
- Boş metin kontrolü

Bu işlemler `src/preprocessing.py` dosyasında yer alıyor.

---

## Kullanılan Teknolojiler

Projede kullandığım teknolojiler:

 Teknoloji  Kullanım Amacı 

 Python  Projenin ana programlama dili 
 pandas  Veri setini okuma ve düzenleme 
 NumPy  Sayısal işlemler 
 scikit-learn  Model eğitimi ve metrik hesaplama 
 NLTK  Metin işleme adımları 
 Streamlit  Web arayüzü 
 joblib  Eğitilen modeli kaydetme 
 matplotlib  Grafik ve görsel çıktı oluşturma 
 GitHub  Proje takibi ve teslim 

---

## Model Yapısı

Projede şu modeli kullandım:

```text
TF-IDF Vectorizer + Logistic Regression
```

TF-IDF yöntemi, haber metinlerini sayısal vektörlere dönüştürmek için kullanıldı. Logistic Regression modeli ise bu vektörlere göre haberin sahte mi gerçek mi olduğunu tahmin etti.

Bu modeli seçmemin nedeni, metin sınıflandırma problemleri için basit, hızlı ve anlaşılır bir yöntem olmasıdır.

---

## Model Performansı

Model eğitildikten sonra başarısını şu metriklerle değerlendirdim:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

Model eğitiminden sonra terminalde performans sonuçları görüntüleniyor. Böylece modelin sahte ve gerçek haberleri ayırt etme başarısını kontrol edebildim.

---

## Dosyaların Görevleri

### `src/preprocessing.py`

Bu dosyada metin temizleme işlemleri bulunuyor. Hem model eğitimi sırasında hem de kullanıcıdan gelen yeni haber metinleri için bu dosyadaki temizleme işlemleri kullanılıyor.

### `src/train_model.py`

Bu dosya model eğitim sürecini içeriyor. `Fake.csv` ve `True.csv` dosyalarını okuyup veriyi hazırlıyor, metinleri temizliyor, TF-IDF dönüşümü yapıyor ve Logistic Regression modelini eğitiyor.

### `src/predict.py`

Bu dosya eğitilmiş modeli kullanarak yeni haber metinleri için tahmin yapıyor. Kullanıcıdan gelen metni temizliyor, modele gönderiyor ve sonucu döndürüyor.

### `src/app.py`

Bu dosyada Streamlit arayüzü bulunuyor. Kullanıcı buradan haber metni girebiliyor, `.txt` dosyası yükleyebiliyor, örnek haberleri seçebiliyor ve tahmin sonucunu görebiliyor.

---

## Karşılaştığım Durumlar

Projeyi geliştirirken fark ettiğim en önemli noktalardan biri, modelin İngilizce veri setiyle eğitildiği için Türkçe haberlerde her zaman güvenilir sonuç vermemesiydi.

İlk testlerde Türkçe doğru haber örneklerinin bazen sahte haber olarak sınıflandırıldığını gördüm. Bunun sebebinin modelin Türkçe metinlerle eğitilmemiş olması olduğunu değerlendirdim. Bu nedenle uygulamadaki güvenilir testleri İngilizce haber örnekleri üzerinden yaptım.

Bu durum bana veri seti dili ile test edilen metnin dilinin uyumlu olmasının önemli olduğunu gösterdi.

---

## Önemli Not

Bu model İngilizce haber veri setiyle eğitilmiştir. Bu nedenle İngilizce haber metinlerinde daha sağlıklı sonuç verir.

Türkçe haber metinleriyle yapılan testler sadece demo amaçlı değerlendirilmelidir.

Model haberin doğruluğunu internette araştırmaz. Sadece eğitim verisinden öğrendiği kelime kullanımları ve metin örüntülerine göre tahmin yapar.

---

## Geliştirilebilecek Yönler

Bu proje daha sonra şu yönlerden geliştirilebilir:

- Türkçe haber veri seti eklenebilir.
- Model Türkçe haberler üzerinde yeniden eğitilebilir.
- Daha büyük veri setleri kullanılabilir.
- BERT veya benzeri transformer tabanlı modeller denenebilir.
- Arayüz daha gelişmiş hale getirilebilir.
- Haber kaynağı analizi eklenebilir.
- Girilen haberin internetteki kaynaklarla karşılaştırılması sağlanabilir.
- Farklı makine öğrenmesi modelleri karşılaştırılabilir.

---

## Proje Durumu

Bu projede şu aşamaları tamamladım:

- GitHub reposunu oluşturdum.
- Proposal dokümanını hazırladım.
- Proje klasör yapısını oluşturdum.
- Veri setini projeye uygun şekilde kullandım.
- Metin ön işleme adımlarını hazırladım.
- TF-IDF ve Logistic Regression ile modeli eğittim.
- Streamlit arayüzünü oluşturdum.
- Sahte ve gerçek haber örnekleriyle test yaptım.
- Ekran görüntülerini aldım.
- Teslim için gerekli dosyaları düzenledim.

---

## Geliştirici

**Mustafa Bahadır Emek**  
Bilişim Sistemleri ve Teknolojileri

GitHub repo linki:

```text
https://github.com/BahadirEmek/Sahte_Haber_Tespit
```