# İleri Yapay Zeka WarmUP Ödev 2

Enes Can · 20 Ekim 2026

Bu çalışma, FedFST kaynak kodunu Google Colab'da çalıştırır; Cora, Citeseer, Actor,
Minesweeper ve Squirrel üzerinde POWER ve FedSPA ile karşılaştırır.

[Google Colab defteri](https://colab.research.google.com/drive/1qHrH63OOxe9LtjzCw__zTgEOJPsuzIdN)

Ana teslim dosyası `Ileri_YZ_Odev2_EnesCan_FedFST_POWER_FedSPA.ipynb` dosyasıdır.
Kurulum ve uyarlama kodları, gerçek eğitim çıktıları, doğruluk matrisleri,
AA ve AF karşılaştırmaları ile grafikler notebook'un içindedir.
Colab'da **Çalışma zamanı → Tümünü çalıştır** seçilerek tekrar yürütülebilir.
Defter tüm kodu içerir; bu klasördeki Python dosyalarını ayrıca yüklemek gerekmez.

## Deney kapsamı

- Beş gerçek veri seti ve üç yöntem ile 15 deney.
- Louvain ile üç istemci, sınıf bazında yaklaşık %60/%20/%20 bölme.
- Ortak bölmeler, sabit tohum 24, görev başına 10 tur ve tur başına 3 yerel adım.
- Bütün sınıflar korunur; son eksik sınıf grubu atılmaz.
- Minesweeper tek görevli olduğu için bu veri setinde AF tanımsızdır.
- FedSPA'nın özgün model ve birleştirme kodu aynı görev akışına uyarlanmıştır.
- Yardımcı eğitim bütçeleri azaltılmıştır. Bu deney makale tablolarının birebir
  yeniden üretimi veya istatistiksel üstünlük iddiası değildir.

Kaynak sürümleri, gerekli yamalar ve deneyin sınırları notebook'ta açıklanmıştır.
POWER'ın GitHub bağlantısı erişilemediğinden makalenin resmî
[Zenodo arşivi](https://zenodo.org/records/15487655) kullanılır.

## Dosyalar

- `prepare_sources.py`: sabit kaynak sürümlerini indirme ve çalıştırılabilirlik yamaları.
- `experiment_common.py`: veri setleri, görevler, bölmeler ve metrikler.
- `run_experiment.py`: özgün yöntem sınıflarını kullanan deney çalıştırıcısı.
- `results_summary.csv`: Colab koşusunun özet metrikleri.
- `FedFST_Odev2_deney_kanitlari.zip`: ayrıntılı loglar, ham metrikler ve ortam kaydı.

Öğretim üyelerine paylaşım yetkisi öğrenci tarafından ayrıca verilecektir.
Üst dizindeki özgün FedFST README dosyası kaynak projenin açıklamasıdır.
