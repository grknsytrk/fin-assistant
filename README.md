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

Frontend, `VITE_API_BASE_URL` üzerinden backend adresine bağlanır. Redis yalnızca cache ve kilitleme amacıyla kullanılır; kalıcı veri kaynağı Supabase'tir. Redis kullanılamazsa backend memory cache'e düşebilir.

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
npm --prefix frontend ci && npm --prefix frontend run build && npx wrangler deploy
```

Backend Space güncellenirken yalnızca uygulama dosyaları ve bağımlılıklar yüklenir; runtime SQLite, WAL/SHM, cache ve log dosyaları deploy edilmez.

Canlı servislerde gizli değişkenler platformların secret/variable alanlarında tutulmalıdır:

- Hugging Face: `RAGFIN_DATABASE_URL`, `RAGFIN_REDIS_URL` ve KAP/API anahtarları secret olarak.
- Hugging Face: `RAGFIN_CACHE_BACKEND=redis` variable olarak.
- Hugging Face: `RAGFIN_ADMIN_REFRESH_TOKEN` secret olarak.
- Cloudflare: `VITE_API_BASE_URL` build variable olarak.
- GitHub Actions: `FIN_API_ADMIN_TOKEN` secret'ı, HF secret'ındaki değerle aynı olmalıdır. Bu token yalnızca `Authorization: Bearer ...` ile fon snapshot workflow'unda kullanılır; frontend'e aktarılmaz.

Redis bağlantı URL'si, Supabase database şifresi ve API token'ları GitHub'a, frontend bundle'ına veya README'ye yazılmamalıdır.

## Performans ve cache

Fon, KAP ve piyasa endpoint'lerinde önce mevcut cache okunur; pahalı dış kaynak yenilemeleri cache miss durumunda yapılır. Upstash Redis kullanıldığında birden fazla backend instance'ı aynı cache'i paylaşır ve aynı verinin eşzamanlı olarak tekrar çekilmesi önlenir. Frontend tarafında da sayfa geçişlerinde tekrar istekleri azaltan cache-first/stale-while-revalidate akışı kullanılır.

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

KAP yorumunu etkinleştirmek için `NVIDIA_API_KEY` ve ilgili `NVIDIA_AI_*` değişkenlerini doldurabilirsiniz. Yorum özelliği isteğe bağlıdır; yapılandırılmış KAP tablolarının çalışması için gerekli değildir.

## Test ve doğrulama

```powershell
python -m compileall app src
pytest
git diff --check
```

Frontend için `npm run build` çalıştırın. Runtime SQLite WAL/SHM, KAP cache'leri ve diğer çalışma çıktıları kaynak kodun parçası değildir.
