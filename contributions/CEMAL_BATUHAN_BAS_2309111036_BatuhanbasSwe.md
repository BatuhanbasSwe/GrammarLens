# Cemal Batuhan Baş · Transformer Modelleri ve Repo Sahipliği

- **Öğrenci numarası:** 2309111036
- **GitHub:** [BatuhanbasSwe](https://github.com/BatuhanbasSwe)

## 1. Yapılan çalışmalar ve sorumluluklar

Projede önceden eğitilmiş iki Transformer modelinin bizim veri setimizle ince ayarından (fine-tuning) ve GitHub reposunun sahipliğinden sorumluyum. DistilBERT baseline, DeBERTa-v3 ise projenin yeni yöntemidir.

Tamamladığım çalışmalar:

- GitHub reposunu public olarak açtım ve ekip üyelerini collaborator olarak ekledim.
- İki modeli de aynı kodla eğiten tek bir script yazdım (`src/train_transformer.py`). Modeller arasında yalnızca Hugging Face checkpoint adı ve varsayılan öğrenme hızı değişir; böylece iki model arasındaki fark koddan değil modelden gelir.
- Erken doğrulama olarak DistilBERT'i tek seed ile yalnızca dev setinde eğittim. Dev macro-F1 0.8882 çıktı; prototipteki en iyi klasik sonuç olan 0.49 açıkça geçildi.
- DeBERTa-v3 eğitiminde ilk adımda bütün ağırlıkların NaN olduğu sayısal kararlılık sorununu teşhis edip kökünden çözdüm (ayrıntılar 3. ve 7. bölümde).
- İki modeli 13, 42 ve 2026 seed'leriyle üçer kez eğittim; test ve BLiMP sonuçlarını Görkem'in ortak ölçüm koduyla 2.3'teki formatta kaydettim.
- DeBERTa-v3 tahmin dosyasını Mert'in hata analizi için `results/preds/` altına koydum.
- Bütün sonuç dosyalarımın Görkem'in benchmark doğrulamasından hatasız geçtiğini kontrol ettim.

## 2. İlgili kod dosyaları

| Dosya | Ne işe yarar |
| --- | --- |
| `src/train_transformer.py` | DistilBERT veya DeBERTa-v3'ü train setinde ince ayarlar, epoch'u dev macro-F1 ile seçer; yalnızca `--final` ile test ve BLiMP ölçümü yapıp sonuçları kaydeder. |
| `results/distilbert_seed{13,42,2026}.json` | DistilBERT'in üç seed için ölçülmüş test, BLiMP, süre ve donanım sonuçları. |
| `results/deberta-v3_seed{13,42,2026}.json` | DeBERTa-v3'ün üç seed için ölçülmüş test, BLiMP, süre ve donanım sonuçları. |
| `results/preds/distilbert_seed*.csv`, `results/preds/deberta-v3_seed*.csv` | Test setindeki 2.114 örnek için `id,label,pred` satırları. |

Script, Görkem'in `src/metrics.py` (`save_predictions`, `save_result_json`, `CLASS_ORDER`) ve `src/evaluate_blimp.py` (`evaluate_blimp`) fonksiyonlarını değiştirmeden kullanır.

## 3. Kodun çalışma mantığı

### Genel akış

```text
train.csv ──tokenize──> eğitim döngüsü (3 epoch) ──her epoch──> dev macro-F1
                                                     │
                              en iyi epoch'un ağırlıkları (RAM'de)
                                                     │
                                  --final verildiyse ▼
                     test tahminleri + BLiMP ──> save_predictions / save_result_json
```

### Modellerin tanımı

`MODELS` sözlüğü, sonuç dosyalarında kullanılan model adını Hugging Face checkpoint'ine ve varsayılan öğrenme hızına eşler:

| `--model` | Checkpoint | Varsayılan LR |
| --- | --- | --- |
| `distilbert` | `distilbert/distilbert-base-uncased` | 3e-5 |
| `deberta-v3` | `microsoft/deberta-v3-base` | 2e-5 |

`AutoModelForSequenceClassification`, önceden eğitilmiş gövdenin üzerine 7 çıkışlı yeni bir sınıflandırma katmanı ekler. İnce ayarda bu yeni katman ve bütün gövde birlikte eğitilir.

### Tekrarlanabilirlik (`set_seed`)

Python, NumPy ve PyTorch rastgele üreteçleri seed'e bağlanır; cuDNN deterministik moda alınır ve `torch.use_deterministic_algorithms` açılır. Eğitim verisinin karıştırılma sırası da aynı seed'li bir `torch.Generator` ile belirlenir. Sonuç olarak aynı seed ile yapılan iki koşu aynı dev skorlarını verir (5. bölümde doğrulandı).

### Tokenizasyon ve dinamik padding (`encode`, `make_collate`)

Cümleler modelin kendi tokenizer'ı ile alt-kelime parçalarına bölünür ve en fazla 64 token'da kesilir. `encode` ayrıca 64 token'ı aşan cümle sayısını raporlar; train, dev ve testte bu sayı **0**'dır, yani hiçbir cümle kesilmemiştir.

Cümleler önceden 64'e doldurulmaz. `make_collate` her batch'i yalnızca o batch'teki en uzun cümle kadar doldurur (dinamik padding). Sonuç değişmez ama hesaplama ciddi şekilde azalır.

### Eğitim döngüsü (`train`)

- **Optimizer:** AdamW, weight decay 0.01. Bias ve LayerNorm ağırlıklarına weight decay uygulanmaz.
- **Öğrenme hızı planı:** Toplam adımların ilk %10'unda 0'dan hedef LR'ye doğrusal ısınma (warmup), sonra 0'a doğrusal azalma. Önceden eğitilmiş ağırlıkların ilk adımlarda büyük güncellemelerle bozulmasını önler.
- **Gradient clipping:** Gradyan normu 1.0 ile sınırlanır.
- **Karışık hassasiyet:** İleri hesap RTX 4070'in desteklediği bf16 ile `torch.autocast` içinde yapılır; loss ve softmax `float()` ile 32 bitte hesaplanır.
- **Sayısal kararlılık kontrolü:** Her adımda loss sonlu değilse (NaN/inf) eğitim `FloatingPointError` ile hemen durur; bozuk bir model sessizce eğitilmeye devam etmez.
- **Epoch seçimi:** Her epoch sonunda dev setinde macro-F1 hesaplanır. En yüksek skoru veren epoch'un ağırlıkları CPU belleğinde kopyalanır ve eğitim bitince modele geri yüklenir. Ağırlıklar diske yazılmaz.

### fp32 ana ağırlıklar

Model `from_pretrained(..., dtype=torch.float32)` ile yüklenir. Transformers 5, bir checkpoint'i kaydedildiği veri türünde yükler; `microsoft/deberta-v3-base` fp16 olarak kayıtlıdır. Ağırlıklar fp16 olduğunda AdamW'deki `eps = 1e-8` fp16'nın gösterebildiği en küçük değerin altında kaldığı için 0'a yuvarlanır, gradyanın karesi de 0'a iner ve güncelleme 0/0 = NaN olur. Bu satır, ağırlıkları ve optimizer durumunu fp32'de tutar; bf16 yalnızca ileri hesapta kullanılır. DistilBERT zaten fp32 kayıtlı olduğu için onun için bir şey değişmez ve iki model aynı kodla eğitilmeye devam eder.

### Final ölçüm (`final_test`)

Test seti yalnızca `--final` verildiğinde ve eğitim bittikten sonra okunur. Test tahminleri `save_predictions` ile, bütün metrikler `save_result_json` ile kaydedilir. BLiMP için `evaluate_blimp`'e cümle listesini alıp 7 sınıfın softmax olasılıklarını döndüren bir fonksiyon verilir; `CORRECT` olasılığı bu softmax çıktısından gelir. Olasılıklar satır toplamı tam 1 olsun diye 64 bitte hesaplanır.

## 4. Algoritmalar, yöntemler ve kütüphaneler

| Araç veya yöntem | Kullanım amacı |
| --- | --- |
| Transformer mimarisi | Self-attention ile cümledeki her token'ı diğer bütün token'larla ilişkilendiren ağ yapısı |
| DistilBERT (`distilbert-base-uncased`) | BERT'ten bilgi damıtma (knowledge distillation) ile elde edilmiş 6 katmanlı, küçük harfli baseline model; 67,0M parametre |
| DeBERTa-v3 (`deberta-v3-base`) | Kelime içeriğini ve konumunu ayrı vektörlerle hesaplayan disentangled attention ve ELECTRA tarzı ön eğitim kullanan 12 katmanlı model; 184,4M parametre (98,4M'si 128.100 token'lık sözlüğün embedding'i) |
| Fine-tuning | Önceden eğitilmiş modeli 7 sınıflı görevimizle uçtan uca yeniden eğitmek |
| AdamW, linear warmup/decay, gradient clipping | Kararlı ince ayar için standart optimizasyon ayarları |
| bf16 autocast + fp32 ana ağırlıklar | Hız için karışık hassasiyet, kararlılık için 32 bit ağırlıklar |
| PyTorch 2.14.1 (CUDA) | Model eğitimi |
| Hugging Face Transformers 5.19.0 | Model, tokenizer ve LR planlayıcı |
| scikit-learn 1.9.1, pandas 2.3.3 | Dev macro-F1 hesabı ve CSV okuma |
| `sentencepiece` | DeBERTa-v3 tokenizer'ı için |

## 5. Çalıştırma ve test

Kurulum (PyTorch'un CUDA'lı sürümü önce kurulmalıdır):

```bash
python -m pip install -r requirements.txt
```

Test setini okumayan geliştirme koşusu:

```bash
python src/train_transformer.py --model distilbert --seed 42
python src/train_transformer.py --model deberta-v3 --seed 42
```

Final koşular (her model için 3 seed):

```bash
python src/train_transformer.py --model distilbert --seed 13 --final
python src/train_transformer.py --model distilbert --seed 42 --final
python src/train_transformer.py --model distilbert --seed 2026 --final
python src/train_transformer.py --model deberta-v3 --seed 13 --final
python src/train_transformer.py --model deberta-v3 --seed 42 --final
python src/train_transformer.py --model deberta-v3 --seed 2026 --final
```

Diğer parametreler (`--epochs`, `--lr`, `--batch_size`, `--max_len`, `--warmup_ratio`, `--weight_decay`, `--max_grad_norm`, `--precision {bf16,fp32}`) varsayılan değerleriyle kullanıldı.

Yapılan kontroller:

- İki tokenizer'ın da yüklendiği ve örnek cümleyi beklenen şekilde böldüğü kontrol edildi.
- Train, dev ve testte 64 token'ı aşan cümle olmadığı ölçüldü (0 / 0 / 0).
- DeBERTa-v3 NaN sorunu, aynı seed ve aynı batch sırasıyla adım adım tekrar üretildi; düzeltmeden sonra 4 adım boyunca loss, gradyan ve ağırlıkların sonlu kaldığı doğrulandı.
- **Tekrarlanabilirlik:** Seed 42'nin yalnızca dev koşusu ile final koşusu, iki modelde de her epoch'ta aynı dev macro-F1 değerlerini verdi (DistilBERT 0.7856 / 0.8808 / 0.8882, DeBERTa-v3 0.9387 / 0.9501 / 0.9516).
- Her tahmin CSV'sinin 2.114 satır olduğu kontrol edildi.
- Görkem'in `build_benchmark()` fonksiyonu, çıktıları repo dışındaki geçici bir klasöre yazdırılarak bütün sonuç dosyalarıyla çalıştırıldı. Fonksiyon her JSON'daki metrikleri tahmin CSV'sinden yeniden hesaplayıp karşılaştırır; hatasız tamamlandı.

## 6. Deneyler ve sonuçlar

### Kullanılan ayarlar

Görev dosyasındaki başlangıç ayarları kullanıldı; dev sonuçları iyi olduğu için hiçbir ayar değiştirilmedi.

| Ayar | Değer |
| --- | --- |
| En fazla token | 64 |
| Batch | 32 |
| Epoch | 3 (en iyi epoch dev macro-F1 ile seçildi) |
| Öğrenme hızı | DistilBERT 3e-5, DeBERTa-v3 2e-5 |
| Warmup | Adımların %10'u, sonra doğrusal azalma |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Hassasiyet | bf16 autocast, fp32 ağırlıklar |
| Seed'ler | 13, 42, 2026 |
| Donanım | NVIDIA GeForce RTX 4070 Laptop GPU (8 GB) |

### Erken doğrulama (yalnızca dev, seed 42)

| Model | Epoch 1 | Epoch 2 | Epoch 3 |
| --- | ---: | ---: | ---: |
| DistilBERT | 0.7856 | 0.8808 | **0.8882** |
| DeBERTa-v3 | 0.9387 | 0.9501 | **0.9516** |

### Seed bazında final sonuçlar

| Model | Seed | En iyi epoch | Dev macro-F1 | Test macro-F1 | Accuracy | Macro P | Macro R | BLiMP çift | BLiMP tür | Eğitim süresi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DistilBERT | 13 | 3 | 0.8891 | 0.8871 | 0.8884 | 0.8866 | 0.8884 | 0.9354 | 0.8940 | 172 sn |
| DistilBERT | 42 | 3 | 0.8882 | 0.8916 | 0.8926 | 0.8918 | 0.8926 | 0.9495 | 0.9104 | 181 sn |
| DistilBERT | 2026 | 3 | 0.8930 | 0.8924 | 0.8936 | 0.8918 | 0.8936 | 0.9433 | 0.9075 | 149 sn |
| DeBERTa-v3 | 13 | 2 | 0.9523 | 0.9530 | 0.9532 | 0.9529 | 0.9532 | 0.9827 | 0.9608 | 447 sn |
| DeBERTa-v3 | 42 | 3 | 0.9516 | 0.9522 | 0.9522 | 0.9526 | 0.9522 | 0.9834 | 0.9615 | 668 sn |
| DeBERTa-v3 | 2026 | 2 | 0.9511 | 0.9503 | 0.9503 | 0.9504 | 0.9503 | 0.9820 | 0.9588 | 501 sn |

### Üç seed özeti

Ortalama ve sapma Görkem'in benchmark koduyla aynı yöntemle (popülasyon standart sapması) hesaplandı; güven aralıkları `build_benchmark.py` ile hesaplanan seed ortalaması üzerinden 1.000 tekrarlı bootstrap aralığıdır. Resmî tablo `results/benchmark.csv` dosyasıdır.

| Metrik | DistilBERT | DeBERTa-v3 |
| --- | ---: | ---: |
| Test macro-F1 (ort. ± sapma) | 0.8904 ± 0.0023 | **0.9518 ± 0.0011** |
| Macro-F1 %95 güven aralığı | [0.8780, 0.9019] | [0.9436, 0.9596] |
| Accuracy | 0.8915 | 0.9519 |
| Macro precision | 0.8901 | 0.9520 |
| Macro recall | 0.8915 | 0.9519 |
| BLiMP çift doğruluğu: genel / SVA / NOUN_NUM | 0.943 / 0.898 / 0.976 | 0.983 / 0.977 / 0.987 |
| BLiMP tür doğruluğu: genel / SVA / NOUN_NUM | 0.904 / 0.839 / 0.952 | 0.960 / 0.955 / 0.964 |

İki modelin güven aralıkları çakışmamaktadır.

### Sınıf bazında F1 (üç seed ortalaması ± sapma)

| Sınıf | DistilBERT | DeBERTa-v3 | Fark |
| --- | ---: | ---: | ---: |
| `CORRECT` | 0.720 ± 0.003 | 0.873 ± 0.005 | +0.153 |
| `SVA` | 0.943 ± 0.006 | 0.978 ± 0.002 | +0.035 |
| `VERB_FORM` | 0.975 ± 0.001 | 0.986 ± 0.001 | +0.011 |
| `DET` | 0.878 ± 0.005 | 0.931 ± 0.001 | +0.053 |
| `NOUN_NUM` | 0.988 ± 0.002 | 0.993 ± 0.002 | +0.005 |
| `PREP` | 0.772 ± 0.007 | 0.922 ± 0.003 | +0.150 |
| `WORD_ORDER` | 0.958 ± 0.001 | 0.980 ± 0.001 | +0.022 |

### Yorum

- Ön eğitimli iki model de baseline'ları büyük farkla geçti: macro-F1 Naive Bayes'te 0.298, CNN'de 0.547 ± 0.009 (ekip arkadaşlarımın sonuç dosyalarından).
- DeBERTa-v3, DistilBERT'e göre macro-F1'i 0.062 artırdı. En büyük kazanç `CORRECT` ve `PREP` sınıflarındadır; bunlar tek bir kelimenin biçiminden çok cümlenin anlamına bakmayı gerektiren sınıflardır.
- BLiMP'te Naive Bayes ve CNN'in çift doğruluğu şans seviyesi olan 0.5'in altındadır; DistilBERT 0.943, DeBERTa-v3 0.983'e ulaştı. Modeller, kendi scriptimizle üretilmemiş minimal çiftlerde de özne-fiil ve belirleyici-isim uyumunu ayırt edebilmektedir.
- DeBERTa-v3'ün iki seed'inde en iyi epoch 2 oldu; 3. epoch dev skorunu düşürdüğü için otomatik olarak 2. epoch seçildi.
- DeBERTa-v3 seed 42'nin en çok karıştırdığı sınıf çifti `PREP → CORRECT`'tir (101 hatanın 24'ü). Bu, etiket kontrolünde görülen "edat değişse de cümle doğru kalabiliyor" durumuyla uyumludur; ayrıntılı inceleme Mert'in hata analizindedir.

## 7. Sorunlar, eksikler ve iyileştirme önerileri

- **DeBERTa-v3 NaN sorunu:** İlk denemede eğitim ikinci adımda NaN loss ile durdu. Teşhis için adım adım tanı scripti yazıldı: birinci adımda loss ve gradyanlar sonluyken, öğrenme hızı 0 olmasına rağmen güncellemeden sonra 202 ağırlık tensörü NaN oldu. bf16 ve fp32 autocast ile aynı sonucun çıkması sorunun hesaplamada değil ağırlıklarda olduğunu gösterdi; ağırlıkların fp16 yüklendiği tespit edildi. Ağırlıkları fp32 yükleyen tek satırlık değişiklikle çözüldü.
- **Ortam:** Eğitim makinesinde scikit-learn kurulu değildi; `requirements.txt` ile kuruldu. PyTorch'un CUDA'lı sürümü ayrıca kurulmalıdır.
- **Eğitim süreleri:** Aynı donanımda DeBERTa-v3 süreleri 447–668 saniye arasında değişti. Laptop GPU'nun anlık durumuna bağlı olabilir ama nedeni ölçülmedi; süreler kaba karşılaştırma için kullanılmalıdır. CNN farklı bir GPU'da (Tesla T4) eğitildiği için süreler modeller arasında doğrudan karşılaştırılamaz.
- **Hiperparametre araması yapılmadı:** Başlangıç ayarları dev setinde yeterli sonuç verdiği için LR, epoch veya batch taraması yapılmadı. Dev setine bakılarak küçük bir LR taraması sonuçları bir miktar artırabilir.
- **En zayıf sınıflar `CORRECT` ve `PREP`:** Bu sınıflardaki hataların bir kısmı etiket gürültüsünden kaynaklanıyor olabilir; bu ancak hatalı tahminlerin elle incelenmesiyle ayrılabilir.
- **Veri sınırlılıkları:** Hatalar script ile üretildiği ve her cümlede tek hata olduğu için gerçek öğrenci yazılarındaki birden çok hatalı cümlelerde performans ölçülmedi. BLiMP dış testi yalnızca `SVA` ve `NOUN_NUM` sınıflarını kapsar.
- **İyileştirme önerileri:** `deberta-v3-large` gibi daha büyük bir model, DeBERTa-v3'ün yanlış bildiği `SVA` cümlelerinde özne-fiil mesafesine göre hata analizi ve modelin olasılıklarının ne kadar güvenilir olduğunu ölçen kalibrasyon analizi.
