# Açılış hızı ve finansal veri doğruluğu

7 Eylül 2026 — yerel uygulama değişiklikleri; canlıya dağıtım yapılmadı.

## Uygulanan değişiklikler

- Hisse detayı önce 5 çeyrek ister. Özet görünürken 20 çeyreklik geçmiş arkadan tamamlanır. Sayfadan çıkış isteği/polling'i iptal eder. Aynı hisseye dönüşte kısa süreli tarayıcı belleği kullanılır.
- KAP response cache boşsa, şeması uyumlu disk kaydı dış kaynağa gitmeden normalize edilir. Kayıt yoksa hızlı `pending` yanıtı ve nötr yükleniyor göstergesi sunulur. Kullanıcının yeniden denemesi gerekmez; veri hazır olana kadar otomatik kontrol artan aralıklarla (en çok 30 saniye aralık) sürer. Gizli sekmede veri isteği yapılmaz, sayfadan çıkışta kontrol durur.
- Aynı hissenin 5/10/20 çeyrek talepleri ortak yenileme kilidi kullanır. Yenileme mevcut KAP cache kurallarına uyar; hatalı sonuç başarılı cache'in yerine yazılmaz.
- Fiyat ve çarpan sağlayıcıları arka planda paralel çağrılır. Bilanço normalizasyonundaki TCMB çağrısı ilk yanıt yolundan çıkarıldı; TCMB hataları her çeyrekte tekrar denenmez. Enflasyon katsayısı tamamlanmayan disk yanıtında raporlanan değerlerin kullanılabileceği açıklanır.
- GET istekleri en çok iki kez denenir. Varsayılan toplam bekleme bütçesi 15 saniye; daha uzun özel GET timeout'ları en çok 20 saniye. Yanıt gövdesini okuma ve retry beklemesi de bu bütçeye dahildir. Uzun Retry-After ekranı dakikalarca bekletmez. POST/commentary timeout'u korunur.
- Fon listesi/kategorileri paralel alınır. Fon getiri özeti genel bakışta, dağılım verisi genel bakış/dağılım sekmelerinde yüklenir. Geçmiş verilerin mevcut sekme yapısı korunur.
- TTM yalnızca dört ardışık ve geçerli çeyreklik akımdan hesaplanır. Eksik metrik, boşluk, tekrar veya sonsuz değer reddedilir; sıfır korunur. Belirsiz/YTD alanları çeyreklik akıma dönüştürülmez. Backend null sonucu frontend fallback'iyle yeniden üretilmez.
- Banka/sigortada FD/FAVÖK, net borç/FAVÖK, genel borç/özkaynak ve cari oran gösterimleri engellenir; genel marj grafikleri gizlenir. F/K, PD/DD ve veri uygunsa ROE korunur. Sağlayıcı fallback'i bu engeli aşamaz.
- Fiyat tarihi için bilanço çekilme zamanına düşen fallback kaldırıldı.

## Ölçüm

`scripts/benchmark_kap_opening.py`, 20 dönemlik sentetik ve mevcut şemaya uygun bir disk kaydıyla çalışır. Gerçek sağlayıcılar çağrılmaz. Bellek cache kullanılır, arka plana iş bırakma taklit edilir. Her senaryo 12 kez ölçülür; p95 en yakın sıra yöntemiyle 12. örnektir. Ölçüm doğrudan Python yanıt üretimi içindir; HTTP, Redis ağı, tarayıcı çizimi ve sunucu açılış süresi dahil değildir.

| Senaryo | p50 | p95 |
|---|---:|---:|
| Önceki cache-miss akışı; kontrollü 150 ms üretici | 150,83 ms | 151,50 ms |
| Disk kaydından yeni yanıt | 7,34 ms | 10,77 ms |
| Hazır bellek cache yanıtı | 0,01 ms | 0,05 ms |
| Veri yokken pending yanıtı | 0,04 ms | 0,11 ms |

İlk denemede normalizasyonun TCMB'ye gidebildiği bulundu; bu çağrı çıkarıldıktan sonra yukarıdaki ölçüm alındı. Kontrollü 150 ms baz çizgisi gerçek üretim gecikmesi değildir; bu sayılardan canlı site hızlanma yüzdesi çıkarılamaz.

Yerel mevcut KAP kayıtlarında güncel şema 15'e uyumlu kayıt bulunmadı. Bu kayıtlar sessizce güncel kabul edilmez; ilk kullanımda yeniden hazırlanmaları gerekir. Canlı ortamın disk/Redis durumu ayrıca doğrulanmalıdır.

## Doğrulama

Odaklı backend testleri: TTM tamlığı, sektör oranları, sağlayıcı beklemeyen disk yanıtı, farklı derinliklerde tek yenileme, dış kaynağa gitmeyen normalizasyon ve mevcut KAP cache davranışları.

Frontend testleri: TTM/sıfır/eksik veri, sektör gösterimi, retry bütçesi, iptal ve 5 çeyrek gösterilirken 20 çeyreğin arkadan yüklenmesi. `npm run build` ile TypeScript ve üretim derlemesi doğrulanır.

Canlıya alındıktan sonra hisse/fon listesi ve detayında ilk açılış ile tekrar açılış için tarayıcı ağ kayıtları ve ilk kullanılabilir içerik süresi ölçülmeli. Backend KAP cache/süre logu ve tarayıcı `stock-detail-first-data` performans ölçümü eklendi. Gerçek uçtan uca gecikme henüz ölçülmedi.
