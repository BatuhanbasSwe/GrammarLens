# Veri Hazırlama Belgesi (GrammarLens-EWT)

Bu belge, GrammarLens veri setinin nasıl üretildiğini anlatır: kaynaklar, temizleme kuralları, hata üretme yöntemi, dengeleme ve bölme. Tüm sayılar `scripts/build_dataset.py` çıktısından ve `data/processed/stats.json` dosyasından alınmıştır.

## 1. Veri kaynakları, sürümleri ve lisansları

| Kaynak | Kullanım | Sabit sürüm | Lisans |
| --- | --- | --- | --- |
| [UD English-EWT](https://github.com/UniversalDependencies/UD_English-EWT) | Ana veri: 16.622 gerçek İngilizce cümle, her kelime için dilbilgisi etiketli | commit `4a4d77f` | CC BY-SA 4.0 |
| [BLiMP](https://github.com/alexwarstadt/blimp) | Dış test: 14 dosya, 14.000 doğru/hatalı cümle çifti | commit `3e56b06` | CC BY 4.0 |

- EWT dosyaları: `en_ewt-ud-train.conllu` (12.544 cümle), `en_ewt-ud-dev.conllu` (2.001), `en_ewt-ud-test.conllu` (2.077) ve `LICENSE.txt`.
- BLiMP'ten sadece bizim sınıflarımıza uyan 14 dosya alındı: `NOUN_NUM` için 8, `SVA` için 6.
- Ham dosyalar `scripts/download_data.py` ile indirilir ve `data/raw/` altına konur. Dosyaların değişmediği `data/raw/checksums.txt` ile doğrulanır.
- EWT'den türeyen veri seti CC BY-SA 4.0 lisansıyla paylaşılır.

## 2. Temizleme kuralları

`scripts/build_dataset.py`, cümleleri sırayla (train, dev, test) okur ve her cümleye aşağıdaki kuralları sırayla uygular. Bir kurala takılan cümle atılır ve sonraki kurallara bakılmaz.

| # | Kural | Nasıl uygulanır | Neden | Atılan cümle |
| --- | --- | --- | --- | --- |
| 1 | Veri setinin işaretlediği yazım hatası | Herhangi bir kelimede `Typo=Yes`, `CorrectForm` veya `CorrectSpaceAfter` işareti varsa | "Doğru" etiketli cümleler gerçekten doğru olmalı | 1.347 |
| 2 | Uzunluk | Kelime sayısı 5'ten az veya 40'tan fazla | Kısalar cümle değil, uzunlar birden çok cümle gibi | 3.438 |
| 3 | Link veya e-posta | Metinde `http://`, `https://`, `www.`, `@` veya `.com` geçiyorsa | Gürültü | 209 |
| 4 | Büyük harf | Harflerin %50'sinden fazlası büyük harf | Gürültü | 130 |
| 5 | Çekimli fiil yok | Hiçbir kelimede `VerbForm=Fin` yok | Tam cümle değil | 1.127 |
| 6 | Tekrar eden cümle | Küçük harfe çevrilmiş metin daha önce görüldü (önceki split'ler dahil) | Eğitim ve test arasında sızıntıyı önlemek | 237 |
| | **Toplam atılan** | | | **6.488** |
| | **Kalan** | 16.622 − 6.488 | | **10.134** |

Not: Tekrar kontrolü split'ler arasında da çalışır. Bir cümle train'de görüldüyse dev veya test'te tekrar çıktığında atılır.

## 3. Hata üretme

Her temiz cümleden bir `CORRECT` örnek (cümlenin kendisi) alınır. Ayrıca her hata türünden, cümlede uygun bir yer varsa, **tek hatalı** bir versiyon üretilir. Hatalar EWT'nin hazır etiketleri (kelime türü, kök, bağlı olduğu kelime) kullanılarak üretilir. Uygun yer yoksa o cümleden o hata türü üretilmez. Rastgele seçimler `seed=42` ile sabittir, yani script her çalıştırmada aynı veriyi üretir.

Bozulan kelime olmayan kısımlar, ilk kelimedeki büyük/küçük harf düzeltmesi dışında aynen kalır. Tek dönüşüm ile üretilen metin, aslından farklı değilse örnek atılır.

| Sınıf | Hata türü | Nasıl üretilir | Örnek |
| --- | --- | --- | --- |
| `CORRECT` | Hata yok | Cümlenin kendisi | I am a preferred provider with most insurance companies. |
| `SVA` | Özne ile fiil uyuşmuyor | Görünür öznesi olan bir fiilin tekil/çoğul hali değiştirilir (runs↔run, is↔are, was↔were) | He **are** concerned about the allocation … |
| `VERB_FORM` | Fiilin yanlış hali | Modal veya "to" sonrası fiil, ya da yardımcı fiilli (have/be) fiil yanlış çekime çevrilir (can go → can goes, has gone → has went, is going → is go) | I want to **went** all over … |
| `DET` | Artikel hatası | Rastgele %50 ihtimalle "a/an" silinir, %50 ihtimalle a↔an değiştirilir | I recently took **an** rescue puppy … |
| `NOUN_NUM` | Tekil-çoğul uyumsuz | Tekil artikelli (this, a, every …) çoğul yapılır veya çoğul belirleyicili (these, many, two …) tekil yapılır. Özneler hariç tutulur, çünkü o durumda hata aynı zamanda SVA olurdu | We are planning **this events** for Thursday … |
| `PREP` | Yanlış edat | Bir edat (in, on, at, to, for, of, with, from, by, about) listeden rastgele başka bir edatla değiştirilir | … being given directly **with** John … |
| `WORD_ORDER` | İki kelime yer değiştirmiş | Artikel veya sıfat ile hemen arkasındaki isim yer değiştirir | Later they kept the same name for **company the**. |

Kelime sıralaması hatasında kelimeler aynı kaldığı için `WORD_ORDER` cümlelerinin kelime torbası, doğru halleriyle birebir aynıdır. Bu yüzden sadece kelime sayısına bakan modeller (Naive Bayes) bu sınıfı ayırt edemez.

## 4. Dengeleme ve bölme

- **Bölme:** EWT'nin kendi train/dev/test ayrımı korunur. Yeni bir rastgele bölme yapılmaz. Bir cümlenin bütün versiyonları (doğru ve hatalı hali) aynı split'te kalır, bu yüzden aynı cümle hem train'de hem test'te bulunmaz. `sent_id` sütunu bunu izlemeyi sağlar.
- **Dengeleme:** Her split'in içinde 7 sınıf, o split'teki en küçük sınıfın boyutuna indirilir. Fazla örnekler `seed=42` ile rastgele atılır. En küçük sınıf her split'te `NOUN_NUM`'dur.
- **Karıştırma:** Her split rastgele sıralanır ve `id` (`train-00000`, `dev-00000`, `test-00000` biçiminde) verilir.

### Dengelemeden önceki sınıf sayıları

| Sınıf | train | dev | test |
| --- | --- | --- | --- |
| `CORRECT` | 7.960 | 1.086 | 1.088 |
| `SVA` | 5.021 | 705 | 674 |
| `VERB_FORM` | 4.056 | 487 | 514 |
| `DET` | 2.395 | 312 | 303 |
| `NOUN_NUM` | 2.287 | 311 | 302 |
| `PREP` | 5.191 | 629 | 638 |
| `WORD_ORDER` | 5.580 | 713 | 705 |

### Dengelemeden sonraki sınıf sayıları

| Sınıf | train | dev | test |
| --- | --- | --- | --- |
| Her sınıf (7 sınıf) | 2.287 | 311 | 302 |
| Split toplamı | 16.009 | 2.177 | 2.114 |

Toplam: **20.300 örnek** (7 sınıf × 2.900).

### Dosya formatı

`data/processed/train.csv`, `dev.csv`, `test.csv` sütunları: `id`, `sent_id`, `label`, `text`, `source`. `source` aynı cümlenin hatasız halidir ve sadece analiz içindir, model girdisi değildir. Sayılar ve ayarlar `data/processed/stats.json` içinde de bulunur.

## 5. Etiket kontrolü

Test setinden her sınıftan 20 örnek (toplam 140) elle incelenir: cümle gerçekten etiketindeki gibi hatalı mı? Ayrıntılar `results/label_audit.csv` dosyasındadır.

**Sonuç: henüz yapılmadı.** Kontrol tamamlanınca sınıf başına "hatalı etiket" oranı bu bölüme eklenecektir.

Beklenen risk: `PREP` sınıfında edat rastgele değiştirildiği için cümle bazen hâlâ dilbilgisel olarak doğru kalabilir. Bu oran saklanmadan raporlanacaktır.

## 6. Sınırlılıklar

- Hatalar gerçek öğrenci hatası değil, script tarafından üretilmiştir. Gerçek hatalar daha çeşitlidir.
- Her cümlede en fazla tek bir hata vardır. Birden çok hata içeren cümleler veri setinde yoktur.
- `NOUN_NUM` hatası öznelerde üretilmez.
- Hatalı cümleler gerçek cümlelerin bozulmuş halidir. `CORRECT` ise orijinal metindir.
- `PREP` sınıfında etiket gürültüsü olabilir (Bölüm 5).

## 7. Yeniden üretme

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/build_dataset.py
```

Çıktı: `data/raw/` (ham veri ve `checksums.txt`) ve `data/processed/` (`train.csv`, `dev.csv`, `test.csv`, `stats.json`). Beklenen sonuç: 10.134 temiz cümle ve 20.300 örnek.
