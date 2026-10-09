# Görkem Veysel Göksu · Naive Bayes, Ortak Ölçüm ve Benchmark

- **Öğrenci numarası:** 2309111084
- **GitHub:** [GorkemGoksu](https://github.com/GorkemGoksu)

## 1. Yapılan çalışmalar ve sorumluluklar

Projede bütün modellerin aynı sınıf sırası ve sonuç şemasıyla değerlendirilmesini sağlayan ortak ölçüm altyapısından, BLiMP değerlendirmesinden, Naive Bayes baseline modelinden ve final benchmark üretiminden sorumluyum.

Tamamladığım çalışmalar:

- Yedi sınıflı test tahminlerinden accuracy, macro precision, macro recall, macro-F1, sınıf bazında F1 ve karmaşıklık matrisi hesaplayan ortak ölçüm kodunu yazdım.
- Sonuç JSON'larının ve `id,label,pred` tahmin CSV'lerinin proje formatında, atomik olarak kaydedilmesini sağladım.
- Macro-F1 için 1.000 tekrarlı yüzde 95 bootstrap güven aralığı fonksiyonunu yazdım.
- Projede seçilen 14 BLiMP paradigmasını okuyup çift doğruluğu ve tür doğruluğu hesaplayan modelden bağımsız değerlendirme kodunu yazdım.
- Yalnızca train verisine fit edilen unigram `CountVectorizer` ve `MultinomialNB` baseline modelini eğittim; dev, test ve BLiMP ölçümlerini yaptım.
- Sonuç JSON'larıyla tahmin CSV'lerini karşılıklı doğrulayan, bütün seed'leri özetleyen ve iki SVG grafik üreten benchmark altyapısını hazırladım.
- Naive Bayes'in kelime sırasını neden göremediğini ölçmek için `WORD_ORDER` örneklerinin kelime torbalarını kaynak cümleleriyle karşılaştırdım.

## 2. İlgili kod dosyaları

| Dosya | Ne işe yarar |
| --- | --- |
| `src/metrics.py` | Ortak sınıf sırasını tanımlar; sınıflandırma metriklerini, tahmin CSV'sini, sonuç JSON'unu ve bootstrap güven aralığını üretir. |
| `src/evaluate_blimp.py` | 14 BLiMP JSONL dosyasını doğrular; modelin olasılık çıktılarından çift ve tür doğruluğunu hesaplar. |
| `src/train_nb.py` | Unigram kelime sayılarıyla Multinomial Naive Bayes'i eğitir; dev ölçümü yapar ve yalnızca `--final` ile test/BLiMP sonuçlarını kaydeder. |
| `src/build_benchmark.py` | Tüm model ve seed dosyalarını doğrular; `benchmark.csv` ile Macro-F1 ve BLiMP grafiklerini üretir. Eksik koşu varsa final çıktı yazmaz. |
| `results/nb_seed42.json` | Naive Bayes'in ölçülmüş test ve BLiMP sonuçlarını içerir. |
| `results/preds/nb_seed42.csv` | Test setindeki 2.114 örnek için gerçek etiket ve Naive Bayes tahminini içerir. |

## 3. Kodun çalışma mantığı

### Ortak metrikler

`metrics.py` içindeki `CLASS_ORDER`, bütün modeller için şu sabit sırayı kullanır:

`CORRECT, SVA, VERB_FORM, DET, NOUN_NUM, PREP, WORD_ORDER`

`compute_classification_metrics()` gerçek ve tahmin edilen etiketleri doğruladıktan sonra bütün metrikleri bu sıra ile hesaplar. Bir model herhangi bir sınıfı hiç tahmin etmese bile yedi sınıfın tamamı hesaplamaya katılır ve sıfıra bölme durumunda skor sıfır kabul edilir. Karmaşıklık matrisinde satırlar gerçek sınıfı, sütunlar tahmin edilen sınıfı gösterir.

`save_predictions()` tam olarak `id,label,pred` sütunlarını yazar. `save_result_json()` metrikleri yeniden hesaplar, BLiMP yapısını doğrular ve yalnızca tanımlanan alanları sonuç dosyasına ekler. Her iki fonksiyon da eksik, yinelenen veya bilinmeyen değerleri kabul etmez.

`bootstrap_macro_f1_ci()` test örneklerinden yerine koyarak 1.000 kez yeniden örnek alır. Her örneklemde yedi sınıflı macro-F1 tekrar hesaplanır; dağılımın yüzde 2,5 ve yüzde 97,5 noktaları yüzde 95 güven aralığı olarak döndürülür.

### BLiMP değerlendirmesi

`evaluate_blimp.py`, sekiz determiner-noun agreement dosyasını `NOUN_NUM`, altı subject-verb/distractor agreement dosyasını `SVA` sınıfına eşler. Her dosyanın 1.000 benzersiz minimal çift içermesi zorunludur; toplam 14.000 çift doğrulanır.

Modelden, verilen cümleler için yedi sınıfın olasılıklarını döndüren bir `predict_proba` fonksiyonu alınır. Modelin olasılık sütun sırası ayrıca verildiği için scikit-learn ve Transformer modellerinin farklı sınıf sıraları güvenli biçimde ortak sıraya çevrilir.

- **Çift doğruluğu:** Doğru cümlenin `CORRECT` olasılığı hatalı cümleninkinden kesin olarak yüksekse başarılıdır. Eşitlik başarısız sayılır.
- **Tür doğruluğu:** Hatalı cümlenin en yüksek olasılıklı sınıfı eşlenen `SVA` veya `NOUN_NUM` sınıfıysa başarılıdır.

Bu tanım, tam cümle dil modeli olasılığını kullanan özgün BLiMP ölçümünün projenin yedi sınıflı sınıflandırıcısına uyarlanmış biçimidir.

### Naive Bayes

`train_nb.py`, CSV şemasını ve etiketleri doğrular. `CountVectorizer` yalnızca `train.csv` metinlerinde fit edilir ve tek kelimelik unigram sayıları üretir. Dev, test ve BLiMP cümlelerinde sözlük yeniden öğrenilmez; yalnızca `transform()` kullanılır. Böylece değerlendirme verisinden eğitim verisine bilgi sızmaz.

`MultinomialNB`, varsayılan `alpha=1.0` Laplace yumuşatmasıyla eğitilir. Model deterministik olduğu için proje kuralına uygun olarak yalnızca seed 42 etiketiyle tek final koşusu kaydedilir. Normal çalıştırma sadece dev sonucunu gösterir. Test ve BLiMP ancak açıkça `--final` verildiğinde okunur ve sonuç dosyaları yazılır.

### Benchmark

`build_benchmark.py`, NB için seed 42'yi; CNN, DistilBERT ve DeBERTa-v3 için 13, 42 ve 2026 seed'lerini zorunlu tutar. Her JSON'un metrikleri eşleşen tahmin CSV'sinden yeniden hesaplanır. Tahmin satırları farklı sırada olsa bile `id` üzerinden test setiyle hizalanır; eksik, fazla veya yinelenen kimlikler reddedilir.

Çok-seed modellerde ortalama Macro-F1 ve seed'ler arası popülasyon standart sapması hesaplanır. Bootstrap sırasında aynı test örneği indeksleri bütün seed'lere uygulanır ve her tekrarda seed skorlarının ortalaması alınır. İki grafik ek görselleştirme bağımlılığı gerektirmeyen SVG biçiminde üretilir.

## 4. Algoritmalar, yöntemler ve kütüphaneler

| Araç veya yöntem | Kullanım amacı |
| --- | --- |
| Python | Veri doğrulama, eğitim ve çıktı üretimi |
| pandas | İşlenmiş train/dev/test CSV'lerini okuma |
| NumPy | Olasılık matrisleri, bootstrap örneklemesi ve nicel işlemler |
| scikit-learn `CountVectorizer` | Cümleleri unigram kelime sayısı vektörlerine dönüştürme |
| scikit-learn `MultinomialNB` | Naive Bayes baseline sınıflandırıcısı |
| scikit-learn metrics | Accuracy, precision, recall, F1 ve karmaşıklık matrisi |
| Bootstrap | Macro-F1 için test örneklemi belirsizliğini ölçme |
| JSONL/CSV/JSON | BLiMP girdileri, tahminler ve ortak sonuç formatı |
| SVG | Ek çizim kütüphanesi olmadan yeniden üretilebilir benchmark grafikleri |

## 5. Çalıştırma ve test

Bağımlılıkların kurulması:

```bash
python -m pip install -r requirements.txt
```

Test setini okumayan geliştirme koşusu:

```bash
python src/train_nb.py
```

Tek nihai Naive Bayes koşusu:

```bash
python src/train_nb.py --final
```

Bu komut aşağıdaki dosyaları üretir:

```text
results/nb_seed42.json
results/preds/nb_seed42.csv
```

Bütün ekip sonuçları geldikten sonra benchmark ve grafiklerin üretilmesi:

```bash
python src/build_benchmark.py
```

Bu komut bütün zorunlu JSON ve CSV dosyalarını önce doğrular. Girdiler tam ve tutarlıysa şu çıktıları üretir:

```text
results/benchmark.csv
results/figures/macro_f1.svg
results/figures/blimp_pair_accuracy.svg
```

Yapılan kontroller:

- Ortak sınıf sırası, eksik tahmin edilen sınıf ve 7×7 karmaşıklık matrisi test edildi.
- JSON ve CSV alanlarının tam sırası/şeması doğrulandı.
- Bootstrap sonucunun aynı random seed ile tekrar üretilebildiği doğrulandı.
- 14 BLiMP dosyasının tamamı, dosya başına 1.000 ve toplam 14.000 çift olarak doğrulandı.
- BLiMP olasılık sütunlarının farklı model sınıf sıralarından ortak sıraya doğru çevrildiği test edildi.
- Eşit `CORRECT` olasılıklarının çift doğruluğunda başarısız sayıldığı test edildi.
- Benchmark kodunun eksik ekip sonuçlarında hiçbir yarım tablo veya grafik yazmadığı doğrulandı.
- Naive Bayes JSON metrikleri, 2.114 satırlık tahmin CSV'sinden yeniden hesaplanarak doğrulandı.

## 6. Deneyler ve sonuçlar

### Naive Bayes test ve BLiMP sonuçları

| Metrik | Değer |
| --- | ---: |
| Test accuracy | 0.3098 |
| Test macro precision | 0.3009 |
| Test macro recall | 0.3098 |
| Test macro-F1 | 0.2983 |
| Macro-F1 yüzde 95 bootstrap güven aralığı | [0.2785, 0.3177] |
| BLiMP çift doğruluğu, genel | 0.4442 |
| BLiMP çift doğruluğu, SVA | 0.4740 |
| BLiMP çift doğruluğu, NOUN_NUM | 0.4219 |
| BLiMP tür doğruluğu, genel | 0.2681 |
| BLiMP tür doğruluğu, SVA | 0.1625 |
| BLiMP tür doğruluğu, NOUN_NUM | 0.3474 |
| Eğitim süresi | 0.307 saniye |
| Donanım | CPU, Intel64 Family 6 Model 186 |

### Sınıf bazında F1

| Sınıf | F1 |
| --- | ---: |
| `CORRECT` | 0.1270 |
| `SVA` | 0.3450 |
| `VERB_FORM` | 0.3486 |
| `DET` | 0.3994 |
| `NOUN_NUM` | 0.4219 |
| `PREP` | 0.2495 |
| `WORD_ORDER` | 0.1968 |

Dev setinde macro-F1 0.2935 olarak ölçüldü. Final test macro-F1 sonucu 0.2983'tür; benchmarkta kullanılan değer test sonucudur.

### Kelime torbası körlüğü

Train, dev ve test bölümlerindeki toplam 2.900 `WORD_ORDER` örneğinin 2.899'unda, `CountVectorizer` analizörünün gördüğü kelime sayımları kaynak cümleyle birebir aynıdır: **2.899 / 2.900 = yüzde 99,97**.

Tek istisna `train-09432` satırıdır. Kaynakta `modern m16`, bozulmuş metinde `M modern16` bulunduğu için iki token birleşmiştir. Veri dosyası başka ekip üyesinin sorumluluğunda olduğundan elle değiştirilmemiştir.

Naive Bayes kelimelerin sayısını görüp sırasını görmediği için `WORD_ORDER` F1'i 0.1968'de kalmıştır. Bu sonuç, kelime sırası bilgisinin bu hata türü için gerekli olduğunu gösterir.

CNN, DistilBERT ve DeBERTa-v3 sonuçları henüz repository içinde bulunmadığından final model karşılaştırması, `results/benchmark.csv` ve iki final grafik henüz üretilmemiştir. Eksik sonuçlar ölçülmüş gibi raporlanmamıştır.

## 7. Sorunlar, eksikler ve iyileştirme önerileri

- Final benchmark için CNN'in üç, DistilBERT'in üç ve DeBERTa-v3'ün üç seed sonucunun hem JSON hem tahmin CSV'si olarak gelmesi beklenmektedir.
- Naive Bayes sözcük sırasını temsil etmediği için özellikle `WORD_ORDER` ve `CORRECT` sınıflarında zayıftır. Bu, baseline modelin beklenen yapısal sınırlılığıdır.
- BLiMP çift doğruluğunun 0.5'in altında olması, NB'nin doğru cümleye sistematik biçimde daha yüksek `CORRECT` olasılığı veremediğini gösterir.
- İleride unigram baseline korunarak ayrı deneyler halinde bigram, karakter n-gramı veya TF-IDF denenebilir. Bunlar mevcut saf kelime-sayımı baseline sonucunun yerine geçirilmemelidir.
- Benchmark kodu yalnızca tam sonuç kümesinde çıktı üretir. Sonuç dondurma saatinde eksik kalan deneyler varsa eksik oldukları açıkça raporlanmalı, sahte veya kopyalanmış değer eklenmemelidir.
