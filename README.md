---
title: Fin API
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: hf_entrypoint.py
python_version: "3.11"
---

# RAG-FIN

RAG-FIN, BIST şirketleri, KAP finansalları, TEFAS fonları ve piyasa verilerini tek bir React + FastAPI uygulamasında sunar.

Uygulama yapılandırılmış veri akışlarıyla çalışır:

- KAP bildirimleri ve finansal tablolar KAP cache'lerinden okunur.
- Fon verileri `tefasfon` üzerinden alınır ve SQLite/cache katmanında tutulur.
- Fon geçmişi ve portföy dağılımı fon detay ekranında gösterilir.
- KAP overview yorumu, yalnızca frontend'in gönderdiği yapılandırılmış finansal verileri kullanır.

PDF ingest/indexleme, ChromaDB, sentence-transformers ve doğal dil rapor soru-cevap katmanı bu projede bulunmaz.

## Canlı mimari

Canlı ortamda frontend ve backend ayrı servisler olarak çalışır:

```text
Tarayıcı
   │
   ├── Cloudflare Worker (React/Vite statik frontend)
   │       └── HF Space üzerinde çalışan FastAPI backend
   │               ├── Supabase PostgreSQL (kalıcı veri)
   │               ├── Upstash Redis (paylaşımlı cache ve distributed lock)
   │               └── TEFAS / KAP / piyasa veri kaynakları
```

Canlı adresler:

- Frontend: <https://fin-assistant.soyturkgurkan-61.workers.dev>
- Backend: <https://grknsytrk-fin-api.hf.space>

Frontend, `VITE_API_BASE_URL` üzerinden API adresine bağlanır. Üretim Cloudflare build'inde bu değer `/api` olmalıdır; Worker API isteklerini Hugging Face backend'ine aktarır. Yerel Vite geliştirmesinde varsayılan adres `http://localhost:8000` olarak kalır. Redis yalnızca cache ve kilitleme amacıyla kullanılır; kalıcı veri kaynağı Supabase'tir. Redis kullanılamazsa backend memory cache'e düşebilir.

Piyasa metadata'sı (`/market/universe` ve `/market/stocks/search`) referans verisinden okunur ve canlı fiyat çağırmaz. Fiyat endpoint'i (`/market/stocks`) Redis'te fresh/stale quote cache ve single-flight lock kullanır; upstream geçici olarak erişilemezse son sağlıklı fiyat boş listeyle değiştirilmez.

## Çalıştırma

Geliştirme ortamını hazırlayıp React ve FastAPI servislerini birlikte başlatmak için:

```powershell
.\run.ps1
```

Eski `run_ui.ps1` adı geriye dönük uyumluluk için korunmuştur ve aynı React + FastAPI launcher'ını çağırır; Streamlit başlatmaz.

Backend'i tek başına çalıştırmak için:

```powershell
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Frontend üretim derlemesi:

```powershell
cd frontend
npm install
npm run build
```

## Deploy

Frontend Cloudflare Worker'a, backend ise Hugging Face Spaces'e deploy edilir. Cloudflare build ayarı:

```text
npm --prefix frontend ci && npm --prefix frontend run build && npm --prefix frontend run worker:deploy
```

Cloudflare build variable `VITE_API_BASE_URL` değerini `/api` olarak ayarlayın. `wrangler.toml` yerel/preview geliştirmede cache'i kapalı tutar; üretim deploy'u `wrangler.production.toml` ile yalnızca `GET /api/market/stocks` ve `GET /api/funds` için Workers Cache'i etkinleştirir. Hisse yanıtları seans içinde 3 saniye taze + 30 saniye SWR, seans dışında bir sonraki seansı aşmayan en çok 300 saniye taze; fon listesi 5 dakika taze + 1 saat SWR olarak saklanır. ETag ve `private, no-cache` tarayıcı politikası korunur.

Backend Space güncellenirken yalnızca uygulama dosyaları ve bağımlılıklar yüklenir; runtime SQLite, WAL/SHM, cache ve log dosyaları deploy edilmez.

Canlı servislerde gizli değişkenler platformların secret/variable alanlarında tutulmalıdır:

- Hugging Face: `RAGFIN_DATABASE_URL`, `RAGFIN_REDIS_URL` ve KAP/API anahtarları secret olarak.
- Hugging Face: performans geçmişi için Fintables birincildir; çıkış IP'si Gate tarafından engellenirse `RAGFIN_FINTABLES_PROXY_BASE_URL` ile token korumalı Cloudflare proxy kullanılabilir. Proxy token'ı boş bırakılırsa `RAGFIN_ADMIN_REFRESH_TOKEN` kullanılır.
- Hugging Face: `RAGFIN_CACHE_BACKEND=redis` variable olarak.
- Hugging Face: `RAGFIN_ADMIN_REFRESH_TOKEN` secret olarak.
- Cloudflare: `VITE_API_BASE_URL=/api` build variable olarak.
- Cloudflare Worker: `FIN_API_ADMIN_TOKEN` secret'ı, HF secret'ındaki değerle aynı olmalıdır. KAP Cron Trigger bu token ile yalnızca `POST /admin/kap/refresh` çağrısını yapar; frontend'e aktarılmaz.
- GitHub Actions: aynı token secret'ı manuel KAP refresh fallback'i ve mevcut fon snapshot workflow'u için kullanılabilir.

Redis bağlantı URL'si, Supabase database şifresi ve API token'ları GitHub'a, frontend bundle'ına veya README'ye yazılmamalıdır.

## Performans ve cache

Fon, KAP ve piyasa endpoint'lerinde önce mevcut cache okunur; pahalı dış kaynak yenilemeleri cache miss durumunda yapılır. Upstash Redis kullanıldığında birden fazla backend instance'ı aynı cache'i paylaşır ve aynı verinin eşzamanlı olarak tekrar çekilmesi önlenir. Endeks, döviz, emtia, hisse grafikleri, KAP snapshot ve karşılaştırma serileri fresh/stale zarfı kullanır: stale veri dönerken yalnızca bir worker arka planda yenileme yapar. Frontend tarafında da sayfa geçişlerinde tekrar istekleri azaltan cache-first/stale-while-revalidate akışı kullanılır.

Supabase'teki `ragfin_json_cache` payload'ları için backend süreç içinde 60 saniyelik bir yerel doğrulama penceresi tutar: aynı cache anahtarı bu süre içinde yeniden okunursa Supabase'e gidilmez, yerel kopya kullanılır. Pencere dolduğunda tek bir doğrulama yapılır ve eşzamanlı istekler aynı okumayı paylaşır. `RAGFIN_JSON_CACHE_LOCAL_READ_TTL_SECONDS=0` bu davranışı kapatıp her okumayı Supabase'e gönderir. Yerel isabet, Supabase okuması ve miss/fallback sayıları cache anahtarı veya fon kodu içermeyen toplu log satırlarıyla izlenir; varsayılan 300 saniyelik pencere `RAGFIN_JSON_CACHE_METRICS_FLUSH_SECONDS` ile değiştirilir.

KAP akışı için yenileme ve okuma yolları ayrıdır: Cloudflare Worker Cron Trigger her dakika hızlı `POST /admin/kap/refresh?fast=1`, beş dakikada bir de kategori backfill'i içeren derin `POST /admin/kap/refresh` çağrısını yapar. Backend olayları `ragfin_kap_flow_events` tablosunda kalıcılaştırır ve Redis'e yalnızca hızlı head/bounded fallback kopyasını yazar. Kullanıcı isteği normalde KAP'a gitmez: ilk 50 kayıt keyset cursor ile döner, eski kayıtlar `before` cursor'ıyla sayfalanır, yeni kayıt kontrolü ise `GET /market/flow/head` ile hafifçe yapılır. Aktif akış ekranı bu head endpoint'ini 5 saniyede bir kontrol eder; son başarılı yenileme 30 saniyeden eskiyse backend mevcut veriyi hemen sunup Redis kilidiyle tek bir hızlı arka plan yenilemesi kuyruğa alır. Böylece yeni bir bildirim çoğunlukla aynı dakika içinde görünür; sağlayıcının yayınlama ve ağ gecikmesi nedeniyle aynı saniye garantisi yoktur. Son başarılı yenileme 15 dakikadan eskiyse aynı mekanizma self-heal olarak da çalışır. GitHub Actions workflow'u otomatik scheduler değil, manuel fallback olarak tutulur.

Fonların KAP portföy dağılımında PDF cache'i uzun süre korunur; ancak KAP'ın hafif bildirim listesi 60 saniyede bir yeniden doğrulanır. Yeni `disclosureIndex` görüldüğünde yalnızca yeni PDF indirilip parse edilir. Fonun Genel Bakış veya Portföy Dağılımı ekranı açıkken frontend de holdings yanıtını 60 saniyede bir yeniler; bu nedenle KAP'ın yayınlama/ağ gecikmesi dışında yeni rapor normalde en geç birkaç dakika içinde görünür.

Metin katmanı olmayan KAP PDF sayfaları PDFium ile görüntülenip yerel CPU OCR ile okunur. OCR tablo hücrelerini satırlarına göre birleştirir; ISIN, nominal değer, toplam değer, grup yüzdesi ve toplam yüzdesi bulunan tabloların son yüzde sütunu fon ağırlığı olarak kullanılır. Türkçe ve İngilizce sayı biçimleri desteklenir. İlk OCR okuması daha uzun sürebilir; çıkarılan metin sonraki isteklerde önbellekten kullanılır. Docker ve Hugging Face ortamları için gerekli görüntü kütüphaneleri sırasıyla `Dockerfile` ve `packages.txt` içinde tanımlıdır.

Fon varlık dağılımı özet kartı ile `Fon Dağılımı` geçmişi aynı en yeni TEFAS rapor gününü kullanır. Geçmiş cache'i daha yeni bir gün içeriyorsa bu gün otomatik olarak özet snapshot'ına yükseltilir; özet snapshot'ı eskiyse API aynı 30 günlük geçmiş yenileme işini arka planda başlatır ve frontend sonucu beklemeden mevcut veriyi gösterip tamamlanınca günceller.

Cloudflare Worker Cron kurulumu için önce `FIN_API_ADMIN_TOKEN` secret'ını tanımlayın (`npm --prefix frontend run worker:secret:fin-api-admin-token`); değer Hugging Face `RAGFIN_ADMIN_REFRESH_TOKEN` ile aynı olmalıdır. Ardından `npm --prefix frontend run worker:deploy` çalıştırın. `wrangler.production.toml` içindeki `* * * * *` dakikalık hızlı, `*/5 * * * *` ise derin yenileme tetikleyicisidir; ikisi de UTC'dir. Deploy sonrası Worker > Settings > Triggers ve Workers Logs ekranlarında tetiklemeleri doğrulayın.

Production'da `/health` yanıtındaki `cache_backend` değeri `redis` ve `cache_redis_fallback` değeri `false` olmalıdır. Aksi durumda uygulama yalnızca process-memory cache kullanır; çoklu worker sağlayıcı çağrılarını paylaşamaz.

## API yüzeyi

Temel endpoint'ler:

- `GET /health`
- `GET /funds`
- `GET /funds/categories`
- `GET /funds/{code}`
- `GET /funds/{code}/history`
- `GET /funds/{code}/allocations`
- `GET /kap/snapshot?company={ticker}`
- `GET /kap/companies`
- `POST /kap/overview-commentary`
- `GET /market/universe`
- `GET /market/stocks/search?q={query}`
- `GET /market/stocks`
- `GET /market/flow?limit=50`
- `GET /market/flow?limit=50&before={cursor}`
- `GET /market/flow/head`
- `POST /admin/kap/refresh` (Bearer admin token)
- aktif market ve endeks endpoint'leri

Eski `/stocks/:ticker/ask` adresleri bozulmaz; frontend bu yolu şirketin Genel Bakış sekmesine yönlendirir. RAG'e ait `/ask`, `/ingest`, `/index`, `/stats`, `/commentary` ve `/feedback` endpoint'leri artık sunulmaz.

## Yapılandırma

`config.yaml` yalnızca çalışma verisi yolu ve KAP ayarlarını içerir. Hassas değerler `.env` içinde tutulmalıdır; başlangıç şablonu için `.env.example` dosyasını kopyalayın.

Önemli değişkenler:

- `RAGFIN_KAP_ENABLED`, `RAGFIN_KAP_CACHE_TTL_HOURS`
- `RAGFIN_KAP_API_KEY`, `RAGFIN_KAP_API_SECRET`
- `RAGFIN_KAP_VYK_BASE_URL`, `RAGFIN_KAP_VYK_AUTH_MODE`
- `RAGFIN_TEFAS_FUND_TYPES`, `RAGFIN_TEFAS_OPEN_ONLY`
- `RAGFIN_FUNDS_LIST_MIN_AUM` (boşsa varsayılan filtre yoktur)
- `RAGFIN_CACHE_BACKEND`, `RAGFIN_REDIS_URL`
- `RAGFIN_KAP_FLOW_HEAD_CACHE_TTL_SECONDS`, `RAGFIN_KAP_FLOW_RETAINED_EVENTS`

KAP AI yorumunda IFM kullanılır. `IFM_API_KEY` değerini yerelde proje kökündeki `.env` dosyasına, Hugging Face dağıtımında ise Space'in **Settings → Variables and secrets** bölümüne secret olarak ekleyin. İsteğe bağlı model/endpoint ayarları `IFM_MODEL` ve `IFM_BASE_URL` ile belirlenir; varsayılan model `IFM/K2-Horizon-375B-A23B`, endpoint `https://api.ifm.ai/v1` olur. Anahtar yalnızca backend'de okunur. IFM anahtarı yoksa eski NVIDIA ayarları geriye dönük fallback olarak kullanılabilir. Yorum özelliği isteğe bağlıdır; yapılandırılmış KAP tablolarının çalışması için gerekli değildir.

## Test ve doğrulama

```powershell
python -m compileall app src
pytest
git diff --check
```

Frontend için `npm run build` çalıştırın. Runtime SQLite WAL/SHM, KAP cache'leri ve diğer çalışma çıktıları kaynak kodun parçası değildir.
