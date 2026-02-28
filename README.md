# RAG-Fin (Week-1 ... Week-10)

Yerel (local-first) RAG pipeline: coklu sirket destekli finansal rapor retrieval + grounded cevap + trend/ratio analizi uretir.
Urun katmaninda arayuz adi: **Bilanco Asistani**.

## Week-16: Tek Komut Demo

Deterministic sample dataset repo icinde gelir (`data/demo_bundle/*`).

### En hizli yol

Windows:

```powershell
.\.venv39\Scripts\python.exe -m ragfin.demo
```

macOS / Linux:

```bash
python -m ragfin.demo
```

Alternatif:

```bash
ragfin-demo
ragfin demo
```

Bu komut:
1. `data/demo/` altinda izole demo workspace olusturur
2. sample fixture veri setini ingest eder
3. `index_v2` olusturur
4. Streamlit UI'yi demo config ile acar

### Doctor (ortam kontrolu)

```bash
ragfin-doctor
# veya
python -m ragfin.doctor
# veya
python -m src.cli doctor
```

## Optional LLM Commentary (OpenRouter)

Varsayilan olarak kapali gelir ve urun local-first calismaya devam eder.
Proje kokundeki `.env` dosyasi otomatik yuklenir.

1. `.env.example` dosyasini `.env` olarak kopyalayin ve API key'i ekleyin:

```bash
OPENROUTER_API_KEY=...
RAGFIN_LLM_ASSISTANT_ENABLED=true
```

2. `config.yaml` icinde commentary katmanini acin:

```yaml
llm_assistant:
  enabled: true
  provider: openrouter
  model: arcee-ai/trinity-large-preview:free
  max_tokens: null  # limitsiz (max_tokens gonderilmez)
  timeout_s: 8.0
  temperature: 0.2
  reasoning_enabled: true
```

Davranis:
- LLM sadece cikarilmis yapisal KPI/ratio/delta verisini yorumlar.
- Sayi uretmez, KPI rakamlarini tekrar etmez.
- Hata durumunda guvenli bos yorum doner (uygulama akisi bozulmaz).
- UI'da yorum sadece butonla tetiklenir (`AI Assistanta Sor`).
- Deterministik cevap/extraction her zaman ana dogruluk kaynagidir.
- Model secimi UI `Settings > AI Assistant Modeli` altindan degistirilebilir.

## 60 Saniyede Baslangic

> Not: `chromadb` su an `Python 3.14` ile uyumlu degil. `Python 3.9 - 3.12` kullanin.

```powershell
cd d:\Projeler\rag-fin
py -3.9 -m venv .venv39
.\.venv39\Scripts\python.exe -m pip install --upgrade pip
.\.venv39\Scripts\python.exe -m pip install -r requirements.txt
.\.venv39\Scripts\python.exe -m pip install -e .
.\.venv39\Scripts\python.exe -m src.cli ingest
.\.venv39\Scripts\python.exe -m src.cli index_v2
.\.venv39\Scripts\python.exe -m streamlit run app/ui.py
```

Sonra `Ask` sekmesinde ornek soru:
- `2025 ucuncu ceyrek net kar kac?`

## API Ornegi (curl)

API server:

```powershell
.\.venv39\Scripts\python.exe -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Tek istek:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"2025 ucuncu ceyrek net kar kac?\",\"retriever\":\"v3\",\"mode\":\"single\",\"company\":\"BIM\"}"
```

CSV export:

```bash
curl "http://127.0.0.1:8000/export?type=trend&company=BIM"
```

Feedback kaydi:

```bash
curl -X POST http://127.0.0.1:8000/feedback \
  -H "Content-Type: application/json" \
  -d "{\"company\":\"BIM\",\"quarter\":\"Q3\",\"metric\":\"net_margin\",\"extracted_value\":\"%4,8\",\"user_value\":\"%4,6\",\"evidence_ref\":\"[doc|Q3|12|gelir tablosu]\",\"verdict\":\"yanlis\"}"
```

## Demo Script (5 Soru)

1. `2025 ucuncu ceyrek net kar kac?`
2. `ilk yariyilda FAVOK marji yuzde kac?`
3. `Q1 Q2 Q3 net kar trendi nasil?`
4. `2025 ikinci ceyrek magaza sayisi kac?`
5. `BIM 2025 uzay madenciligi gelir hedefi nedir?` (beklenen: Dokümanda bulunamadı)

## Ekran Goruntusu Placeholder

```text
[PLACEHOLDER] docs/assets/ui_ask_tab.png
```

## Week-8 Dokumanlari

- Diyagramlar: `docs/diagrams/architecture.md`, `docs/diagrams/retrieval_flow.md`
- Trade-off raporu: `docs/reports/tradeoffs.md`
- Error analysis: `docs/reports/error_analysis.md`
- Career kit: `docs/career/cv_bullets.md`, `docs/career/interview_story.md`

## Week-9 Ozeti (Multi-Company Financial Intelligence)

- Ingestion/chunk/index metadata:
  - `company`, `quarter`, `year` alanlari eklendi.
- Retrieval filtreleri:
  - CLI: `--company BIM`
  - API: `/ask` body icinde `"company":"BIM"`
  - company verilmezse tum sirketlerde arama.
- Yeni oran motoru: `src/ratio_engine.py`
  - `net_margin`, `favok_margin`, `revenue_growth_qoq`, `store_growth_qoq`
- Cross-company mode:
  - `karsilastir`, `hangisi daha iyi` veya birden fazla sirket adinda tetiklenir.
- CSV export:
  - API: `GET /export?type=trend&company=BIM`
  - UI: trend ve ratio tablolari icin CSV indirme.
- Streamlit:
  - yeni `Dashboard` sekmesi (sirket secici + KPI kartlari + trend chart).

## Week-10 Ozeti (Human-Friendly Product Layer)

- Smart Upload:
  - PDF adindan sirket/ceyrek/yil otomatik algilama
  - tek adimda otomatik `ingest + index_v2`
- Company Health Dashboard:
  - KPI kartlari + QoQ yon (ok) bilgisi
  - Net marj renk kodlama
  - konfigurasyon bazli `GREEN / YELLOW / RED` health label
- Finansal Ozet:
  - extracted metric + trend verisinden 5 maddelik executive summary
- Change Detection Panel:
  - son 2 ceyrekte iyilesen / kotulesen / yatay kalan KPI listesi
- Clean UI:
  - muhendislik secenekleri varsayilan gizli
  - `Advanced Mode` expander altinda

## Week-12 Ozeti (Coverage Sprint + Auto-Verify + Feedback Loop)

- Yeni audit araci: `src/coverage_audit.py`
  - `python -m src.cli coverage_audit --company MIGROS`
  - metrik bazli coverage/invalid/verified_pass oranlari ve eksik nedenlerini uretir.
- Dictionary suggestion yardimcisi:
  - `python -m src.cli dict_suggest --company SOK`
  - eksik kalan sorularda sayi yakinindaki etiket ifadelerini (top 30) listeler.
- Auto-Verify katmani:
  - extractor secilen metrik icin `verify_status: PASS/WARN/FAIL` uretir.
  - WARN durumlari: aday uyusmazligi, birim belirsizligi, olasi yil uyumsuzlugu.
- Dashboard geri bildirim dongusu:
  - KPI kartlari icin `Dogru/Yanlis` geri bildirim formu.
  - kayit dosyasi: `data/processed/feedback.jsonl`.
- API:
  - `/ask` evidence satirlarinda verify alanlari (`verify_status`, `verify_warnings`).
  - yeni endpoint: `POST /feedback`.

## Kurulum (Windows)

> Not: `chromadb` şu an `Python 3.14` ile uyumlu değildir. `Python 3.9 - 3.12` kullanın.

```powershell
cd d:\Projeler\rag-fin
py -3.9 -m venv .venv39
.\.venv39\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Week-4 Demo UI (Streamlit)

Demo arayuzu:

```powershell
streamlit run app/ui.py
```

Python 3.14 benzeri surumlerden kacmak icin onerilen dogrudan komut:

```powershell
.\.venv39\Scripts\python.exe -m streamlit run app/ui.py
```

Kisa yol:

```powershell
.\run_ui.ps1
```

UI Week-13 ile sidebar tabanli product akisina gecti:
- `Overview (Genel Bakis)`:
  - sirket + donem secimi (`Latest/Q1/Q2/Q3`)
  - KPI kartlari (Net kar, Ciro, FAVOK, marjlar, magaza)
  - Net kar / Net marj trend grafikleri
  - "Bu ceyrekte ne degisti?" paneli
  - mini verify badge + `Guven & Kanit Detayi` expander
- `Companies (Sirketler)`:
  - secilen KPI'nin sirketler arasi donem bazli karsilastirma tablosu
- `Reports (Raporlar)`:
  - smart PDF upload
  - tek adim: `Yeni Dosyalari Ice Al + Indexle`
  - indexed dokuman listesi ve son ingest loglari
- `Ask (Soru Sor)`:
  - sade soru kutusu + tek buton
  - kanitlar varsayilan kapali
  - markdown export
  - retriever secimi sadece `Advanced` altinda
- `Settings (Ayarlar)`:
  - retriever/top-k/alpha/beta gibi muhendislik ayarlari
  - metrics/latency/coverage araclari

UI log dosyasi:
- Her soru loglanir: `data/processed/ui_logs.jsonl`
- Alanlar: `timestamp`, `question`, `retriever`, `parsed`, `top_sources`, `found`

## Week-5 (Cross-Quarter Financial Intelligence Mode)

### Neler eklendi

- `src/metrics_extractor.py`:
  - metrik bazli regex + baglam eslestirme ile numeric ayiklama
  - desteklenen metrikler:
    - `net_kar`
    - `favok`
    - `favok_marji`
    - `brut_kar_marji`
    - `satis_gelirleri`
    - `magaza_sayisi`
  - yapisal cikti:
    - `quarter`, `metric`, `value`, `unit`
    - ek olarak kaynak metadata: `doc_id`, `page`, `section_title`, `chunk_id`, `excerpt`
- Ceyrek agregasyonu (Q1/Q2/Q3):
  - pandas DataFrame uzerinde toplama
  - `abs_change`, `pct_change`, `direction` hesaplama
- Trend-aware answer mode (v4):
  - tetikleyiciler: `trend`, `artis`, `azalis`, `karsilastir`
  - otomatik cross-quarter retrieval + metrik ayiklama
  - UI'da tablo + line chart + degisim ozeti
- Groundedness korunur:
  - metrik bulunamayan ceyrekler `None/Bulunamadi` olarak isaretlenir
  - kaynak kanitlari ve aranan kaynaklar gosterilir

### Week-5 ornek sorular

1. `Q1 Q2 Q3 net kar trendi nasil?`
2. `Q1 Q2 Q3 FAVOK marji trendi nasil?`
3. `Q1 Q2 Q3 brut kar marji karsilastir`
4. `Q1 Q2 Q3 satis gelirlerinde artis var mi?`
5. `Q1 Q2 Q3 magaza sayisi degisimi nedir?`

## Week-1 (Baseline)

### Ingest

```powershell
python -m src.cli ingest
```

Cikti: `data/processed/pages.jsonl`

### Index (v1)

```powershell
python -m src.cli index
```

Collection: `bimas_faaliyet_2025`

### Query (v1)

```powershell
python -m src.cli query "net kâr"
```

Top-5 sonuc alanlari:
- `distance/score`
- `doc_id`, `quarter`, `page`, `chunk_id`
- metnin ilk 400 karakteri

### Grounded answer

```powershell
python -m src.cli ask "2025 3. çeyrekte net kar ne kadar?"
```

## Week-2 (Improved Retrieval)

### Neler degisti

- Heading-aware chunking:
  - sayfada baslik tespiti (ALL CAPS, numarali baslik, finans anahtar kelimeleri)
  - section bazli bolme
- Paragraph-first chunking + overlap
- Table-like block ayrimi (`block_type="table_like"`)
- Zengin metadata:
  - `doc_id`, `quarter`, `page`, `chunk_id`, `section_title`, `block_type`, `chunk_version`
- Yeni v2 collection:
  - `bimas_faaliyet_2025_v2`
- Retrieval iyilestirmesi:
  - once vector top-k (default 15)
  - sonra lexical boost + rerank
  - quarter filtresi (`--quarter Q1/Q2/Q3`)

### Index (v2)

```powershell
python -m src.cli index_v2
```

Ciktilar:
- `data/processed/chunks_v2.jsonl`
- Chroma collection: `bimas_faaliyet_2025_v2`

### Query (v2)

```powershell
python -m src.cli query_v2 "net kâr" --quarter Q3
```

Toplanan alanlar:
- `distance`, `raw_score`, `vector_score`, `lexical_boost`, `final_score`
- `doc_id`, `quarter`, `page`, `chunk_id`, `section_title`, `block_type`
- metin onizlemesi

### V1 vs V2 eval karsilastirma

```powershell
python -m src.cli eval_compare --top-k 5 --top-k-initial-v2 15 --alpha 0.35
```

veya:

```powershell
python eval/run_eval.py --top-k 5 --top-k-initial-v2 15 --alpha 0.35
```

Cikti:
- `data/processed/eval_retrieval_comparison.jsonl`
- soru bazinda:
  - `v1_top_pages`
  - `v2_top_pages`
  - `v2_top_section_titles`

## Week-3 (Measurable Retrieval + Query Parsing + V3)

### Neler eklendi

- Gold-labeled eval dosyasi:
  - `eval/gold_questions.jsonl` (40+ TR soru)
- Otomatik metrik hesaplama:
  - `hit@1`, `hit@3`, `hit@5`, `MRR@5`, `quarter_accuracy@1`
- Query parser (`src/query_parser.py`):
  - Turkce ifadelerden otomatik quarter tespiti (`Q1/Q2/Q3`)
  - soru tipi cikarimi (`numeric`, `trend`, `qualitative`, `kpi`)
- Query-aware retrieval v3:
  - v2 ustune query-type-aware boost
  - quarter auto-filter (sorudan otomatik)
- Tek komutla metrik raporu:
  - v1 vs v2 vs v3

### Query v3

```powershell
python -m src.cli query_v3 "2025 üçüncü çeyrek net kâr kaç?"
```

```powershell
python -m src.cli ask_v3 "riskler neler?"
```

### Metrics report

```powershell
python -m src.cli metrics_report
```

Ciktilar:
- `data/processed/eval_metrics_detailed.jsonl`
- `data/processed/eval_metrics_summary.json`

### Metriklerin anlami

- `hit@1`: dogru eslesme ilk sirada mi
- `hit@3`: dogru eslesme ilk 3 icinde mi
- `hit@5`: dogru eslesme ilk 5 icinde mi
- `MRR@5`: ilk dogru sonucun reciprocal rank ortalamasi
- `quarter_accuracy@1`: top1 sonucun quarter'i beklenen quarter ile uyumlu mu

### Week-3 ornek sorgular (auto quarter + type-aware)

1. `python -m src.cli query_v3 "2025 üçüncü çeyrek net kâr kaç?"`
2. `python -m src.cli query_v3 "ilk yarıyılda FAVÖK marjı yüzde kaç?"`
3. `python -m src.cli query_v3 "dokuz aylık satışlar geçen yıla göre arttı mı?"`
4. `python -m src.cli query_v3 "risk yönetim politikaları nelerdir?"`
5. `python -m src.cli query_v3 "2025 ikinci çeyrek mağaza sayısı kaç?"`
6. `python -m src.cli ask_v3 "2025 üçüncü çeyrek net kâr kaç?"`
7. `python -m src.cli ask_v3 "online satışlarla ilgili bilgi var mı?"`

## Demo Akisi (Ornek)

1. `streamlit run app/ui.py`
2. `Reports` sayfasindan PDF yukleyin.
3. `Yeni Dosyalari Ice Al + Indexle` butonuna basin.
4. `Overview` sayfasinda sirket ve donem secerek KPI'lari gorun.
5. `Ask` sayfasinda soruyu yazip yanit alin.
6. Ihtiyac varsa `Settings` sayfasinda teknik ayarlari acin.

Ornek sorular:
1. `2025 ucuncu ceyrek net kar kac?`
2. `ilk yariyilda FAVOK marji yuzde kac?`
3. `dokuz aylik satislar gecen yila gore artmis mi?`
4. `riskler neler?`
5. `2025 ikinci ceyrek magaza sayisi kac?`
6. `Q1 Q2 Q3 net kar trendi nasil?`
7. `Q1 Q2 Q3 magaza sayisi karsilastir`
8. `BIM 2025 uzay madenciligi gelir hedefi nedir?` (beklenen: bulunamadi)

## Grounded cevap formati

Answer modulu evidence citation formatini su sekilde verir:
- `[doc_id, quarter, page, section_title]`

Kanit bulunamazsa:
- `Dokümanda bulunamadı`
- aranan sayfalar listesi
- retrieval benzerliği düşük veya soru anahtar kelimeleri kanıtlarla örtüşmüyorsa yine `Dokümanda bulunamadı`

## Teknik Notlar

- PDF extraction: `pypdf`
- Bos sayfada fallback: `extraction_mode="layout"`
- Embedding model: `intfloat/multilingual-e5-small`
- Chroma path: `data/processed/chroma`

## Limitasyonlar

- PDF tablo/layout karmaşıklığında extraction kaybi olabilir.
- Table-like heuristic kural tabanlidir; her tabloyu kusursuz ayirmayabilir.
- Lexical boost basit bir katmandir; domain-genisletebilir.
- Rules-based answerer uretken LLM degildir; kanit merkezli calisir.
- Streamlit UI demo odaklidir; cok buyuk dosya setlerinde ilk indexleme CPU uzerinde zaman alabilir.
- Week-5 metrik ayiklama regex+heuristic tabanlidir; PDF tablo bozulmalarinda bazi degerler atlanabilir.

## Week-6 (Retrieval Robustness & Hybrid Benchmark)

### Yeni retrieval modlari

- `v4_bm25`:
  - `rank-bm25` ile `data/processed/chunks_v2.jsonl` corpus uzerinden lexical retrieval
- `v5_hybrid`:
  - vector (v3) + BM25 birlestirme
  - skor: `normalized_vector + beta * normalized_bm25`
- `v6_cross`:
  - v5 adaylari icin cross-encoder rerank
  - model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Week-6 benchmark komutlari

```powershell
python -m src.cli benchmark_week6
```

```powershell
python -m src.cli error_report
```

### Week-6 ciktilari

- `data/processed/eval_metrics_week6.json`
- `data/processed/eval_metrics_detailed.jsonl`
- `data/processed/error_analysis.jsonl`

## Week-7 (Hardening + Packaging)

### Konfigürasyon

- Tek merkezi config dosyasi: `config.yaml`
- Loader: `src/config.py`
- Kapsam:
  - yollar (`data/raw`, `data/processed`)
  - Chroma dizini + collection adlari
  - chunk parametreleri
  - retrieval parametreleri
  - model adlari (embedding + cross-encoder)

CLI ve Streamlit varsayilanlari bu config'ten okunur; CLI flag'leri ile override edebilirsiniz.
Isterseniz farkli config dosyasi vererek calistirabilirsiniz:

```powershell
python -m src.cli --config config.yaml query_v3 "net kar"
```

### API (FastAPI)

Calistirma:

```powershell
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Endpointler:
- `GET /health`
- `GET /stats`
- `POST /ingest`
- `POST /index` (`{"version":"v1"}` veya `{"version":"v2"}`)
- `POST /ask` (`{"question":"...","retriever":"v1|v2|v3|v5|v6","mode":"single|trend"}`)

### Ornek curl

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/index -H "Content-Type: application/json" -d "{\"version\":\"v2\"}"
```

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"2025 ucuncu ceyrek net kar kac?\",\"retriever\":\"v3\",\"mode\":\"single\"}"
```

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"Q1 Q2 Q3 brut kar marji trendi nasil?\",\"retriever\":\"v6\",\"mode\":\"trend\"}"
```

### Testler

Dev bagimliliklari:

```powershell
pip install -r requirements-dev.txt
```

Calistirma:

```powershell
pytest -q
```

Eklenen testler:
- `tests/test_query_parser.py`
- `tests/test_metrics_extractor.py`
- `tests/test_retriever_smoke.py`
- `tests/test_api.py`

### Latency Benchmark

```powershell
python -m src.cli latency_bench
```

Cikti:
- `data/processed/latency_benchmark.json`

### Docker (Opsiyonel)

API icin:

```powershell
docker build -t rag-fin-api .
docker run --rm -p 8000:8000 rag-fin-api
```
