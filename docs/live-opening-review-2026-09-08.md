# Canlı açılış performansı incelemesi

Ölçümler: 7 Eylül 2026. Rapor: 8 Eylül 2026.

## Sonuç

Öncelik fonların ilk yüklenmesini hızlandırmak. Fon listesi, getiri özeti ve portföy içeriği isteklerinde ilk ölçüm ile tekrar arasında büyük fark var. Hisse tarafında önbellekten gelen API yanıtları tekrar ölçümünde yaklaşık 0,2–0,4 saniye; ancak küçük olması beklenen 5 dönemlik finansal yanıt 20 dönem içeriyor.

## Yöntem ve sınırlar

- Canlı frontend: https://fin-assistant.soyturkgurkan-61.workers.dev
- Canlı backend: https://grknsytrk-fin-api.hf.space
- API uç noktalarına aynı HTTP istemcisiyle iki tur sıralı GET isteği gönderildi. Süre, yanıt gövdesinin alınmasını kapsar; JSON çözümleme dahil değildir. İlk bağlantıda bağlantı kurulum maliyeti de bulunur.
- Üretim önbelleği temizlenmedi. “İlk” ilk ölçülen istek anlamındadır; kesin soğuk önbellek ölçümü değildir. Tarayıcı ve API ziyaretleri birbirinin önbelleğini ısıtmış olabilir.
- İki örnek p95, tipik kullanıcı süresi veya kalıcı performans garantisi vermez. Bu çalışma yük testi değildir.
- Tarayıcıda yapay ağ/CPU yavaşlatması uygulanmadı. Süreler otomasyon eylemi ile DOM gözlemi arasındadır; otomasyon maliyeti içerir, LCP/FCP değildir.
- Tüm API yanıtları HTTP 200 ve gzip idi. Aşağıdaki boyutlar açılmış gövde boyutudur; ağda taşınan sıkıştırılmış boyut ölçülmedi.

## API sonuçları

| İstek | İlk ölçüm | Tekrar |
| --- | ---: | ---: |
| BIMAS finansallar, 5 dönem isteği | 976 ms | 312 ms |
| BIMAS finansallar, 20 dönem isteği | 416 ms | 263 ms |
| BIMAS fiyat | 209 ms | 210 ms |
| BIMAS gün içi grafik | 443 ms | 220 ms |
| Fon listesi | 8.220 ms | 472 ms |
| Fon kategorileri | 222 ms | 255 ms |
| AAL detay | 980 ms | 222 ms |
| AAL yaklaşık 6 aylık geçmiş | 867 ms | 221 ms |
| AAL getiri özeti | 4.622 ms | 228 ms |
| AAL varlık dağılımı | 564 ms | 756 ms |
| AAL portföy içeriği | 12.990 ms | 226 ms |

Ham sonuçlar: [live-api-timing-2026-09-07.json](live-api-timing-2026-09-07.json). Tekrar kullanılabilir ölçüm betiği: [measure_live_opening.py](../scripts/measure_live_opening.py).

Portföy içeriğinin 12,99 saniyesi ilgili API/modül süresidir; fon sayfasının tamamının açılma süresi değildir. Varlık dağılımı tekrar isteği hızlanmadı; her uç noktada aynı davranış yok.

## Tarayıcı gözlemleri

| Eylem | Otomasyon dahil gözlenen süre |
| --- | ---: |
| BIMAS genel bakış yeniden açılışı | 1.077 ms |
| BIMAS finansal tablolar sekmesine geçiş | 3.104 ms |
| AKBNK genel bakış ilk ölçülen ziyaret | 1.997 ms |
| AKBNK genel bakış yeniden açılışı | 1.701 ms |
| Fon listesi yeniden açılışı, DOM'da hazır | 2.155 ms |
| AAL yeniden açılışı, grafik hazır | 1.488 ms |

Sekme geçişi ölçümü özellikle locator/otomasyon maliyetinden etkilenebilir; uygulamanın gerçek çizim süresi olarak yorumlanmamalıdır. AAL ilk tıklama ve DOM gözlemi 4.019 ms üst sınır verdi; snapshot maliyeti nedeniyle doğrudan açılış karşılaştırmasına alınmadı. Fon listesinin ilk ziyaret süresi güvenilir biçimde yakalanamadı.

Fon listesinde DOM üzerinden **2.041 tablo satırı** doğrulandı. Bu, görünmeyen satırların da oluşturulduğunu gösterir; CPU maliyetinin ne kadarını oluşturduğunu bu ölçüm tek başına kanıtlamaz.

## Bulgular ve sonraki uygulama sırası

### 1. Fonlarda ilk isteğin beklemesini azalt

Fon listesi 2.041 kayıt ve 1.332.115 bayt açılmış JSON döndürüyor. İlk/tekrar farkı 8.220/472 ms. `app/api.py` içindeki `_funds_listing_payload`, 45 saniyelik yanıt önbelleği kullanıyor; `get_funds_payload(..., auto_refresh=False)` çağırıyor. Bu nedenle beklemeyi doğrudan TEFAS'a bağlamak doğru değil. Liste oluşturma, snapshot okuma, önbellek erişimi ve serileştirme süreleri ayrı ölçülmeli. Bu dekoratör kullanımında eşzamanlı önbellek kaçırmalarını birleştiren `single_flight` açık değil.

Getiri özeti servisinde senkron TEFAS çağrısı, portföy içeriğinde ise gerektiğinde senkron `refresh_fund_holdings` bulunuyor. Mevcut önbellek dekoratörü kaçırmada işlemi istek üzerinde çalıştırıyor. Bunlar gecikme üretebilen doğrulanmış kod yollarıdır; ölçülen gecikmenin her aşamaya dağılımı henüz bilinmiyor.

İlk uygulama paketi:

- Fon listesi üretiminin aşama sürelerini kaydet; gecikmenin kaynağını ayır.
- Geçerli eski veri varsa hemen döndür, yenilemeyi arka planda tamamla; veri tarihini doğru koru.
- Aynı veriyi isteyen eşzamanlı istekleri tek yenilemede birleştir.
- Önbelleksiz durumda nötr yükleme göster; otomatik takip ile veri geldiğinde güncelle.
- Getiri özeti ve portföy içeriğinin yenilenmesi, kullanılabilir temel fon bilgilerini bekletmesin.

Doğrulama: kontrollü önbellek kaçırma ve eşzamanlı istek senaryolarında yanıtın upstream yenilemeyi beklemediğini; tek arka plan işi oluştuğunu; tamamlandığında yeni verinin göründüğünü kontrol et. Canlı önbelleği temizleyerek test yapma.

### 2. Hisse finansallarını gerçekten küçük yanıtla aç

`max_quarters=5` ve `max_quarters=20` isteklerinin ikisi de 20 dönem ve **605.951 bayt** açılmış gövde döndürdü. İkisi de `shared_hit`. İlk 5 dönem, ardından 20 dönem istemek bu ölçümde veri hacmini azaltmıyor.

İstenen derinlik yanıt hazırlanırken uygulanmalı veya istemci zaten yeterli dönem aldığında ikinci aynı içerik isteğini atlamalı. İlk açılış yanıtını küçültmek için sunucu tarafı sınırlama ayrıca gerekli. TTM için gereken ardışık dönemler ve veri bütünlüğü korunmalı.

### 3. Fon listesinde oluşturulan satır sayısını azalt

2.041 satırın tamamını DOM'a eklemek yerine sayfalama veya görünür satırların oluşturulması uygulanabilir. Arama ve sıralama tüm kayıtlar üzerinde doğru çalışmalı. Bu aşama DOM yükünü azaltır; API'nin ilk yanıt gecikmesini tek başına çözmez.

## Kapsam

Bu aşamada uygulama kodu ve dağıtım değiştirilmedi; ölçüm betiği, ham sonuçlar ve bu rapor hazırlandı. AAL geçmiş yanıtında iki ölçümde de `history_job=queued` görüldü; bu tek başına takılı iş kanıtı değildir ve ayrı iş yaşam döngüsü incelemesine bırakıldı.

## İlk uygulama paketi — 8 Eylül 2026

Ölçümün ardından fon listesi, getiri özeti ve portföy içeriği için arka planda yanıt hazırlama eklendi. İlk veri yoksa API `status=pending` döndürüyor; istemci yükleme görünümünü koruyarak artan aralıklarla sonucu alıyor. Geçerli eski yanıt varsa hemen sunuluyor ve `refresh_pending` üzerinden sessizce yenileniyor. Kaynak tarihleri değiştirilmez. İlk kez alınan verinin upstream hazırlama süresi ortadan kalkmış değildir.

Taze yanıt süreleri listede 45 saniye, getiri özetinde mevcut yapılandırma (varsayılan 15 dakika), fiyatlarla zenginleştirilmiş portföy içeriğinde 15 saniyedir. Eski yanıt listede ilave 15 dakika, getiri ve portföy içeriğinde ilave 24 saat saklanır. Portföyün ayrı canlı fiyat güncelleme yolu korunmuştur.

Süreç başına en fazla dört yanıt yenilemesi çalışır. Aynı anahtarda paylaşılan ve süre boyunca yenilenen kilit, işin yinelenmesini engeller. Hata durumunda eski yanıt korunur ve 30 saniyelik tekrar bekleme uygulanır. Kullanılamayan yanıtlar eski veri yedeğine yazılmaz. İstemci soğuk yanıt takibini 120 saniyeyle sınırlar; mevcut verinin arka plan takibi görünmeyen sayfalarda durur.

`fund_catalog` logları snapshot yükleme, filtreleme, günlük getiri uzlaştırma ve sıralama sürelerini; `response_refresh` logları toplam yanıt hazırlama süresini kaydeder. Bunlar sunucu içindeki aşama süreleridir, ağ aktarımı veya tarayıcı çizim süreleri değildir.

Doğrulama: eşzamanlı soğuk istek, eski veriyi koruma, kaynak tarihi ve kopya izolasyonu, hata sonrası bekleme, başka işçinin kilidi ve işçi kapasitesi testleri geçti. İlgili API önbellek/şema testleri, fon snapshot okuma testi, istemci yükleme/iptal testleri ve görünürlük/arka plan takip testleri geçti. Frontend üretim derlemesi başarılı. İlk test çalıştırmasındaki üç geçici dizin erişim hatası, çalışma alanındaki ayrı test dizini kullanılarak giderildi.

Bu uygulama paketi henüz canlıya dağıtılmadı; yeni canlı süreler ölçülmedi. Hisse yanıtını küçültme ve fon tablosunda görünür satırları oluşturma sonraki paketlerdir.
