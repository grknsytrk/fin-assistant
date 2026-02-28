# RAG-Fin Work Summary (Week-1 + Week-2 + Week-3 + Week-4 + Week-5 + Week-6 + Week-7 + Week-8 + Week-9 + Week-10 + Week-11 + Week-12 + Week-13 + Week-14 + Week-15 + Week-16 + Hotfixes + LLM Commentary Enhancement)

Bu dosya projedeki guncel ilerlemeyi ozetler.

## Genel Durum

- Sistem local-first calisir: `pypdf + sentence-transformers + chromadb + rank-bm25`.
- V1/V2/V3 komutlari ve Streamlit akisi korunmustur.
- Week-5 trend/cross-quarter akisi aktif.
- Week-6 retrieval benchmark modlari aktif.
- Week-7 ile config + API + test + packaging katmani eklendi.
- Week-8 ile career-mode dokumantasyon paketi eklendi.
- Week-9 ile multi-company + ratio + cross-company karsilastirma eklendi.
- Week-10 ile human-friendly urun katmani (Bilanco Asistani) eklendi.
- Week-11 ile extraction stabilization + confidence layer eklendi.
- Week-12 ile coverage audit + auto-verify + feedback loop eklendi.
- Week-13 ile sidebar tabanli product UI (finance website look) eklendi.
- Week-14 ile Extraction Quality 2.0 (table reconstruction + consistency + self-verify + range sanity) eklendi.
- Week-15 ile Trust UX + onboarding odakli urunlestirme (UI-only) iyilestirmeleri eklendi.
- Week-16 ile one-command deterministic demo + packaging + doctor + CI smoke katmani eklendi.
- Week-16 sonu ile OpenRouter tabanli opsiyonel LLM commentary (yorum) katmani ve `.env` auto-load destegi eklendi.
- Son hotfixlerle SOK/Migros dashboard metrik cikarma, negatif deger (zarar) destegi, UI/grafik okunabilirligi, Migros KPI tutarliligi, Ask sayfasi ana cevap gorunurlugu ve guven/feedback UX dili iyilestirildi.
- LLM Commentary Enhancement ile gercek rakamli yorum, sayfa-ici model secici, hata raporlama ve multi-model uyumluluk eklendi.

## LLM Commentary Enhancement

### 1) Gercek Rakamli LLM Yorum

- `src/commentary.py`:
  - Metrik ve oran degerleri artik LLM'e gercek sayilarla gonderiliyor (`"mevcut"` yerine).
  - Delta degerleri de numerik olarak iletiliyor.
  - System prompt guncellendi:
    - Rakam yasagi kaldirildi; LLM'e "payload'daki gercek rakamlari kullan" talimati verildi.
    - Buyuk sayilar icin "milyar TL / milyon TL" formatlama talimati eklendi.
    - Yuzde degerleri ve donem-donem degisim yonu + buyuklugu gosterme talimati eklendi.
    - Turkce yazma zorunlulugu eklendi.
  - Post-processing digit/placeholder filtreleri kaldirildi (`_has_digit_in_output`, `_has_placeholder_artifacts`).
  - `_empty_if_invalid` fonksiyonu gevsetildi: ekstra key'ler tolere ediliyor, sadece beklenen 4 key korunuyor.

### 2) Sayfa-ici AI Model Secici

- `app/ui.py`:
  - **Genel Bakis** sayfasinda "AI Assistanta Sor" butonunun yanina `st.selectbox` ile model secici eklendi.
  - **Soru Sor** sayfasinda da ayni sekilde inline model secici eklendi.
  - Kullanici ayarlara gitmeden dogrudan model degistirebiliyor.
  - Mevcut model listesi:
    - `arcee-ai/trinity-large-preview:free`
    - `stepfun/step-3.5-flash:free`
    - `nvidia/nemotron-3-nano-30b-a3b:free`
    - `meta-llama/llama-3.3-70b-instruct:free`
    - `openai/gpt-oss-120b:free`
    - `qwen/qwen3-235b-a22b-thinking-2507`

### 3) Multi-Model Uyumluluk (Fallback)

- `src/commentary.py`:
  - `response_format: json_object` ve `reasoning` parametreleri desteklemeyen modeller icin otomatik fallback eklendi.
  - Ilk istek basarisiz olursa bu parametreler olmadan yeniden deneniyor.
  - Varsayilan timeout `8s` -> `30s`'ye cikarildi (buyuk modeller icin).

### 4) Hata Raporlama (Error Reporting)

- `src/commentary.py`:
  - Hata durumlarinda `_error` key'i ile detayli hata bilgisi donduruluyor:
    - HTTP hatalari (404, 429, 500 vb.) ve hata mesaji
    - Timeout hatalari
    - JSON parse basarisizliklari (ham yanitin ilk 200 karakteri ile)
    - Format hatalari (beklenen key'ler eksikse)
    - API key tanimsizsa
    - Provider/enabled devre disiysa
  - Basarili yanitlarda `_model` key'i ile kullanilan model adi donduruluyor.
- `app/ui.py`:
  - `_render_commentary_box` guncellendi:
    - Hata varsa kirmizi `st.error()` kutusu ile hata mesaji gosteriliyor.
    - Basarili yanitlarda model adi `Model: ...` olarak gosteriliyor.
    - Icerik + hata ayni anda gosterilebiliyor.
    - Icerik yoksa ve hata da yoksa kutu gizleniyor.

## Week-16 Teslimleri

### 1) Deterministic Demo Data Bundle

- Yeni bundle: `data/demo_bundle/`
  - `pages_fixture.jsonl` (BIM/MIGROS/SOK x Q1/Q2/Q3)
  - `gold_questions_demo.jsonl`
  - `gold_questions_demo_multicompany.jsonl`
  - `demo_questions.txt`
- `src/ingest.py`:
  - fixture ingest akisi eklendi (`ingest_page_fixtures`)
  - ingest ozetleyici yardimcilar eklendi.
- Amac:
  - harici indirme olmadan deterministik demo datasetiyle hizli baslatma.

### 2) One-command Demo Runner

- Yeni modul: `src/demo.py`
  - demo workspace: `data/demo/`
  - demo config: `data/demo/config.demo.yaml`
  - fixture ingest + mevcut page-fixture'dan indexleme
  - Streamlit UI'yi demo config ile acma (`RAGFIN_CONFIG`)
- Komutlar:
  - `python -m ragfin.demo`
  - `ragfin-demo`
  - `ragfin demo`

### 3) Packaging ve Entrypointler

- Yeni dosya: `pyproject.toml`
  - console scripts:
    - `ragfin-ui`
    - `ragfin-api`
    - `ragfin-demo`
    - `ragfin-doctor`
    - `ragfin`
- Yeni package wrapperlari: `ragfin/*`
  - `ragfin/__main__.py`, `ragfin/cli.py`, `ragfin/demo.py`, `ragfin/doctor.py`, `ragfin/ui.py`, `ragfin/api.py`
- Ek olarak:
  - `app/__init__.py`
  - `Makefile` (`demo`, `doctor`, `ui`, `api`, `test`)

### 4) Doctor Komutu (Ortam Kontrolu)

- Yeni modul: `src/doctor.py`
- CLI entegrasyonu:
  - `python -m src.cli doctor`
  - `ragfin-doctor`
- Kontroller:
  - Python surumu (`3.9 - 3.12`)
  - `chromadb` / `sentence-transformers` import
  - embedding model local-load kontrolu
  - path write izni
  - Chroma koleksiyon create/delete smoke
- Cikti:
  - tablo formatinda PASS/WARN/FAIL + net aksiyon onerileri.

### 5) UI First-run Onboarding + Demo Script

- `app/ui.py`:
  - Overview no-index durumunda:
    - `Run Demo` aksiyonu (demo bundle ingest+index)
    - `Raporlar sayfasina git` hizli gecisi
    - `ragfin-demo` komut hint'i
  - Ask sayfasi:
    - `Demo Script (hazir sorular)` butonlari (`demo_questions.txt`).
- `src/config.py`:
  - `RAGFIN_CONFIG` env override destegi ile demo config izolasyonu.

### 6) Test + CI Smoke (Week-16)

- Yeni test: `tests/test_demo_packaging.py`
  - hizli demo index smoke (networksiz)
  - metrics_report coverage smoke (BIM/MIGROS/SOK bos degil)
- Yeni script: `scripts/check_demo_metrics_smoke.py`
  - demo metrics summary dosyasini assertion ile dogrular.
- Yeni CI workflow: `.github/workflows/ci-week16.yml`
  - compileall
  - pytest
  - demo prepare
  - demo metrics_report + smoke check

### 7) Optional LLM Commentary Layer (OpenRouter)

- Yeni moduller:
  - `src/openrouter_client.py`
  - `src/commentary.py`
- Kapsam:
  - LLM sadece cikarilmis yapisal KPI/ratio/delta verisini yorumlar.
  - Sayi uretmez; KPI rakamlarini tekrar etmez; kisa JSON cikti verir.
  - Hata/timeout/invalid JSON durumunda guvenli bos sonuc doner:
    - `{\"headline\":\"\", \"bullets\":[], \"risk_note\":\"\", \"next_question\":\"\"}`
- Model/ayarlar:
  - `config.yaml -> llm_commentary`
  - default model: `openai/gpt-oss-120b:free`
  - varsayilan: `enabled: false` (local-first deterministic davranis korunur)
- UI entegrasyonu (`app/ui.py`):
  - Overview: "Bu ceyrekte ne degisti?" panelinde LLM kisa yorum (enabled ise)
  - Ask: ana cevap altinda `Kisa yorum` kutusu (enabled ise)
- API entegrasyonu (`app/api.py`):
  - `/ask` yanitina `commentary` alani eklendi (enabled + found durumunda).

### 8) .env Auto-load (Developer Experience)

- `src/config.py`:
  - Proje kokundeki `.env` dosyasi otomatik yuklenir.
  - `RAGFIN_CONFIG` dahil env override'lari `.env` icinden de okunur.
- Dokumantasyon:
  - `.env.example` OpenRouter/LLM commentary env alanlariyla guncellendi.
  - `README.md` icine "Optional LLM Commentary (OpenRouter)" bolumu eklendi.
- Not:
  - Artik her terminal acilisinda manuel `$env:` veya `setx` zorunlu degildir; `.env` yeterlidir.

## Week-14 Teslimleri

### 1) Structured Table Reconstruction

- `src/metrics_extractor.py`:
  - table-like tespit guclendirildi (`_looks_like_table_chunk`)
  - satir + kolon yeniden kurulum akisi eklendi (`_structured_table_candidates`)
  - ceyrek kolon hizasina gore sayi secimi yapiliyor (Q1/Q2/Q3)
  - satir bazli reconstruction adaylari ana ranking havuzuna dahil edildi
- `src/chunking.py`:
  - `is_table_like_paragraph` daha agresif hale getirildi

### 2) Cross-Quarter Consistency Validator

- `src/metrics_extractor.py`:
  - `trend_consistency_score` eklendi
  - ceyrekler arasi sapma `%300` ustune cikarsa sonraki gecerli aday deneniyor
  - fallback nedeni `trend_consistency_fallback` olarak reason listesine yaziliyor

### 3) Ratio Self-Verification

- `src/ratio_engine.py`:
  - dogrudan marj (net/favok) ile hesaplanan marj karsilastirmasi eklendi
  - fark `> 10` puan ise dogrudan marj fallback'e aliniyor
  - verify warning:
    - `direct_vs_computed_margin_deviation`
    - uygun durumda `margin_fallback_alt_direct_candidate`

### 4) Company Range Sanity Layer (Config Bazli)

- `config.yaml`:
  - extraction ayarlari genisletildi:
    - `trend_deviation_threshold_pct`
    - `ratio_self_verify_pp_threshold`
    - `expected_ranges` (metric bazli min/max)
- `src/config.py`:
  - yeni extraction alanlari parse + validate ediliyor
- `src/validators.py`:
  - `validate_metric_value(..., expected_range=...)` destegi eklendi
  - config range disi degerler reject ediliyor

### 5) Metrics Report (Before/After)

- `src/metrics.py`:
  - multi-company extraction ozetine before/after accuracy eklendi
  - `coverage_rate_before/after`, `extraction_accuracy_before/after`, `delta`
- `src/cli.py`:
  - `metrics_report` ciktilari before -> after formatina guncellendi

### 6) Testler (Week-14)

- Yeni/gelisen test kapsami:
  - structured table quarter-column alignment
  - cross-quarter consistency fallback
  - config expected-range sanity rejection
  - ratio self-verification fallback
- Toplam test sonucu:
  - `python -m pytest -q` -> `22 passed`

## Week-13 Teslimleri

### 1) Sidebar Navigasyon ve Sayfa Mimarisi

- `app/ui.py` ana akisi yeniden duzenlendi.
- Yeni sayfalar:
  - `Overview (Genel Bakis)`
  - `Companies (Sirketler)`
  - `Reports (Raporlar)`
  - `Ask (Soru Sor)`
  - `Settings (Ayarlar / Advanced)`

### 2) Overview (Finance Dashboard Feel)

- Ust bar:
  - sirket secici
  - donem secici (`Latest`, `Q1`, `Q2`, `Q3`)
- KPI kartlari:
  - Net kar / Net zarar
  - Satis gelirleri
  - FAVOK
  - Net marj
  - FAVOK marji
  - Magaza sayisi
- Alt bolum:
  - Net kar trend grafigi
  - Net marj trend grafigi
  - "Bu ceyrekte ne degisti?" paneli
- Guven/verify:
  - kartlarda kisa badge
  - detaylar `Guven & Kanit Detayi` expander altinda

### 3) Ask Sayfasi Sadelestirme

- Tek soru girisi + tek yanit butonu
- Sonuc tek bir cevap kutusunda gosterilir
- Kanitlar varsayilan kapali expander altinda
- Markdown export korunmustur
- Retriever secimi sadece `Advanced` altinda tutulmustur

### 4) Reports Sayfasi Productized Akis

- Smart upload + tekil birincil aksiyon:
  - `Yeni Dosyalari Ice Al + Indexle`
- Dokuman listesi tablosu:
  - `company`, `year`, `quarter`, `filename`, `pages`, `indexed_at`, `status`
- Advanced:
  - `Reindex v1/v2`
- Son ingest/index kayitlari:
  - `data/processed/ingest_logs.jsonl`
  - UI'da `Son Ice Alma Loglari` expanderi

### 5) Settings (Advanced) Sayfasi

- Muhendislik kontrolleri tek yerde toplandi:
  - varsayilan retriever
  - top-k / alpha / beta parametreleri
  - debug toggle
  - metrics/latency/coverage araclarini UI'dan tetikleme
- Uretim varsayilani:
  - `v3`

## Week-12 Teslimleri

### 1) Coverage-driven Diagnostics

- Yeni modul: `src/coverage_audit.py`
- Yeni CLI:
  - `python -m src.cli coverage_audit --company MIGROS`
  - `python -m src.cli coverage_audit --company SOK`
- Cikti:
  - metrik bazli `coverage_rate`, `invalid_rate`, `verified_pass_rate`
  - eksik metrikler ve olasi nedenler (`label_mismatch`, `unit_parse_or_scaling_fail`, `table_split_or_no_numeric_evidence`)
  - dosya: `data/processed/coverage_audit_<company>.json`

### 2) Dictionary Expansion Workflow

- `data/dictionaries/metrics_tr.yaml` genisletildi:
  - ek synonym ve section hint setleri (company-agnostic)
- Yeni CLI:
  - `python -m src.cli dict_suggest --company SOK`
- `dict_suggest`:
  - missing-case retrieval sonuclarinda sayi cevresindeki etiket ifadelerini (top-N) listeler.

### 3) Auto-Verify Katmani

- Yeni modul: `src/autoverify.py`
- `src/metrics_extractor.py`:
  - secilen aday icin `verify_status: PASS/WARN/FAIL`
  - `verify_checks`, `verify_warnings`, `verify_reasons`
  - uyari sinyalleri:
    - guclu aday uyusmazligi
    - birim belirsizligi
    - olasi yil uyumsuzlugu
    - alternate regex uyusmazligi

### 4) UI + API Surface

- `app/ui.py` (Dashboard):
  - KPI bazinda verify badge (`PASS/WARN/FAIL`)
  - confidence panelinde verify warning detaylari
  - KPI geri bildirim formu (Dogru/Yanlis + duzeltme/Bulunamadi)
  - feedback dosyasi: `data/processed/feedback.jsonl`
- `app/api.py`:
  - `/ask` evidence alanlarinda `verify_status` + `verify_warnings`
  - yeni endpoint: `POST /feedback`

### 5) Metrics Report Genisletmesi

- `src/metrics.py`:
  - multi-company extraction ozetine `verified_pass_rate` eklendi
  - per-company/per-metric coverage + invalid + verified pass oranlari
- `src/cli.py`:
  - `metrics_report` ve `benchmark_week6` ciktilarinda `verified_pass` gosterimi eklendi.

## Week-11 Teslimleri

### 1) Data-driven Metric Dictionary

- Dosya: `data/dictionaries/metrics_tr.yaml`
- Metrikler:
  - `net_kar`, `favok`, `satis_gelirleri`, `magaza_sayisi`
  - `net_kar_marji`, `favok_marji`, `brut_kar_marji`
- Zengin synonym + section hint yapisi ile extractor genellestirildi.

### 2) Candidate Generation + Ranking

- `src/metrics_extractor.py`:
  - Her metrik icin `top-5` aday uretilir (`extract_metric_with_candidates`).
  - Agirlikli skor:
    - label/synonym/fuzzy match
    - line/window proximity
    - `table_like` boost
    - section_title boost
    - quarter/year alignment
    - uygunsuz baglam cezasi (`baz puan`, `%` vb.)
  - En iyi aday sanity check'ten gecmezse otomatik sonraki adaya fallback yapilir.

### 3) Confidence Layer

- Secilen metrik kaydinda:
  - `confidence` (0..1)
  - `reasons[]`
  - `candidates[]` (debug)
- `src/ratio_engine.py`:
  - confidence propagation eklendi (`min` mantigi).
  - `overall_confidence` ve `confidence_map` uretiliyor.

### 4) Validators

- Yeni modul: `src/validators.py`
  - `validate_metric_value(...)`
  - `validate_ratios(...)`
- Asiri/absurt degerler filtreleniyor, makul negatif degerler destekleniyor.

### 5) UI + API Confidence Surface

- `app/ui.py` (Dashboard):
  - low-confidence badge (`config.yaml -> extraction.low_confidence_threshold`)
  - `Why?` paneli (reasons + evidence)
- `app/api.py`:
  - `/ask` yanitinda `answer.confidence`
  - evidence satirlarinda `confidence` (trend/cross-company modunda reasons dahil)

### 6) Multi-Company Evaluation (Extraction)

- Yeni gold dosya: `eval/gold_questions_multicompany.jsonl` (45 soru)
  - `BIM=15`, `MIGROS=15`, `SOK=15`
- `src/metrics.py`:
  - `metrics_report` ozetine per-company extraction coverage/invalid-rate eklendi.
- `src/cli.py`:
  - `metrics_report` ve `benchmark_week6` komutlarina `--multi-company-gold-file` eklendi.

## Week-10 Sonrasi Hotfixler

### 1) Sirket Filtresi Guvenilirligi

- `src/retrieve.py`:
  - Sirket filtresi varken fallback ile baska sirketten sonuc donme davranisi kaldirildi.
  - Sonuc yoksa dogrudan bos donerek groundedlik korundu.

### 2) Migros Dosya Adi/Donem Algisi

- `src/ingest.py`:
  - Ceyrek parser genisletildi.
  - `2c-2025`, `3c-2025`, `2025-3q` benzeri adlar artik `2025Q2/Q3` olarak parse ediliyor.

### 3) Metric Extractor Iyilestirmeleri (Migros Sunum Formati)

- `src/metrics_extractor.py`:
  - Marj metriklerinde (brut/net/FAVOK marji) metin arasi/uzak yuzde formatlari icin regex genisletildi.
  - `ciro` anahtar kelimesi satis gelirleri metrigine baglandi.
  - TL metriklerinde `%` ve `baz puan` baglami kaynakli yanlis yakalamalar filtrelendi.
  - `2024 -> 2025` ikili kolonlu satirlarda guncel yil degerini tercih eden kural eklendi.
  - Migros `9A/3C` sunum desenleri icin net kar/FAVOK/satis metrikleri desteklendi.

### 4) Dashboard Veri Dolulugu ve Okunabilirlik

- `app/ui.py`:
  - KPI kartlari tek satir yerine 2 satira bolundu (`...` truncation azaldi).
  - Parasal formatlar okunur hale getirildi:
    - `mlr TL`, `mn TL`
  - Delta yoksa (`None`) gereksiz/kirmizi isaret cikmiyor.
  - Dashboard icin ratio retrieval daha genis aday havuzuyla calisiyor (`top_k_initial>=30`, `top_k_final>=12`).
  - `Ciro / Satis` karti ve trend grafigi eklendi.

### 5) Ratio Engine Fallback Stratejisi

- `src/ratio_engine.py`:
  - `net_kar_marji` ve `favok_marji` dogrudan cikarimi eklendi.
  - Hesaplanan marj (net_kar/satis) yerine dogrudan bulunan marj varsa onu onceleyen fallback stratejisi uygulandi.

### 6) Query Anlama ve Eslestirme

- `src/query_parser.py`:
  - `ciro`, `hasilat`, `satis` numeric sinyallerine eklendi.
- `src/retrieve.py`:
  - Query expansion'a `ciro -> satis/satislar/hasilat` eslestirmesi eklendi.

### 7) Son Durum (Migros Dashboard)

- Migros icin asagidaki seriler artik dashboardda daha tutarli doluyor:
  - `Net kar (Q1/Q2/Q3)`
  - `Brut kar marji`
  - `FAVOK marji`
  - `Net marj`
  - `Ciro / Satis`
- `revenue_growth` ve `store_growth` alanlari, kaynakta veri yoksa bilincli olarak `-` kalir (grounded).

### 8) SOK Net Marj / Olcekleme Duzeltmesi

- `src/metrics_extractor.py`:
  - Gelir tablosu `table_like` baglaminda implicit milyon olcekleme kurali eklendi.
  - SOK benzeri sunumlarda `67.651` gibi degerlerin yanlis `TL` yorumlanmasi azaltildi.
- `src/validators.py`:
  - `satis_gelirleri` ve `net_kar/favok` icin asiri dusuk TL degerleri olasi olcek hatasi olarak reddedilir.
  - fallback mekanizmasi ile sonraki aday secilir.

### 9) Negatif Deger (Zarar) Destegi

- `src/metrics_extractor.py`:
  - Negatif parse:
    - `-123,4`
    - `(123,4)`
    - baglam sinyali (`zarar`, `net donem zarari`) ile otomatik negatifleme
  - `mn/mlr` unit varyasyonlari desteklendi.
- `src/validators.py`:
  - Negatif marjlar default kabul edilir.
  - Sadece absurt araliklar reddedilir (`< -200%` veya `> 200%`).
- `app/ui.py`:
  - `net_kar < 0` ise kart etiketi `Net zarar`.
  - Delta kurali netlestirildi:
    - ok yerine `Iyilesme/Kotulesme/Yatay` metni kullanilir (delta_color off).

### 10) Migros KPI Tutarlilik Duzeltmesi (Q3 net kar + magaza sayisi)

- `src/metrics_extractor.py`:
  - `magaza_sayisi` icin aday penceresi genisletildi (kucuk/yanlis sayilar yerine toplam magaza adayi daha guclu secilir).
  - Kucuk ama gecerli aday secilirse, daha yuksek ve makul store adayi icin global fallback eklendi.
  - `store_count_total_preferred` reason etiketi eklendi (debug/izlenebilirlik).
- `src/ratio_engine.py`:
  - `build_ratio_table` varsayilan retrieval parametreleri guclendirildi (`top_k_initial=30`, `top_k_final=12`).
  - Ayni guclendirme cross-company karsilastirma akisina da uygulandi.
- `app/ui.py`:
  - Overview/Companies sayfalarinda KPI uretimi icin minimum retrieval havuzu yukseltilerek Q3 verilerinin kacma riski azaltildi.
- Etki:
  - Migros icin `Q3 net kar` dashboardda daha tutarli gorunur.
  - Migros `magaza sayisi` kartinda `56` gibi yanlis kucuk aday yerine toplam magaza degerleri tercih edilir.

### 11) Ask Ana Cevap Multi-Company Stabilizasyonu

- `app/ui.py`:
  - Ask ana cevap cikarimi cok katmanli hale getirildi:
    - `extractor` tabanli secim
    - metrik-odakli retrieval fallback
    - `ratio_engine` tabanli son fallback
  - Amaç:
    - sirketten bagimsiz (`BIM`, `MIGROS`, `SOK`) sayisal cevap gorunurlugunu artirmak
    - "Yanit hazir ama deger yok" durumlarini azaltmak
  - Dusuk bilgi icerikli metinler (yalnizca "kanitlarda bulundu" tipi) filtrelenerek ana cevap karti netlestirildi.

### 12) Ask Ana Cevapta Zarar Renk Semantigi

- `app/ui.py`:
  - Ask ana cevap degeri icin yeni stil chip'i eklendi.
  - Negatif degerler (zarar) artik yesil yerine **kirmizi** gosterilir.
  - Pozitif/notr degerler yesil tonda kalir.

### 13) Overview Grafik Tipi ve Deger Okunabilirligi

- `app/ui.py`:
  - Parasal trend grafiklerinde (`net_kar`, `satis_gelirleri/ciro`, `favok`) gosterim `sutun grafik` olarak guncellendi.
  - Marj metrikleri (`net_marj`, `favok_marji`, `brut_kar_marji`) `cizgi grafik` olarak korunmustur.
  - Grafik deger/tooltip formatlari insan-okur olacak sekilde normalize edildi (ham uzun sayi yerine `mlr TL` / `mn TL` formati).
  - Negatif parasal degerler (zarar) sutun grafiklerde dogru isaret ve olcekle gosterilir.
- Etki:
  - Overview ekraninda ciro/net kar gibi buyuk tutarlar daha hizli ve dogru yorumlanir.
  - Marj trendleri oran mantigina uygun sekilde line chart'ta takip edilir.

### 14) Week-15 Trust UX + Onboarding Productizasyonu (UI-only)

- `app/ui.py`:
  - Overview no-index durumunda onboarding akisina hizli gecis eklendi:
    - `Raporlar sayfasina git` butonu (sidebar sayfa yonlendirmesi ile)
    - mevcut `Load sample report` akisiyla birlikte first-run sureci netlestirildi.
  - Sidebar navigasyon state'i session-state ile senkronlandi:
    - sayfalar arasi yonlendirme onboarding icinden de tutarli calisir.
  - KPI detay panel dili urunlestirildi:
    - baslik/icerik sadeleştirildi (`Nereden Geldi?`).
  - KPI feedback formu kullanici diliyle guncellendi:
    - `Dogru / Yanlis / Emin degilim`
    - `Yanlis` seciminde dogru deger + opsiyonel not alinir.
  - Guven detay panelinde teknik level yerine kullanici etiketi gosterimi yapildi.
- `app/ui_components.py`:
  - Trust badge dili guncellendi:
    - `Yuksek / Orta / Dusuk guven`
  - Dahili seviye mantigi (`High/Medium/Low`) korunarak sadece sunum dili degistirildi.
- Kapsam:
  - Core extraction/retrieval algoritmalarina dokunulmadan yalnizca UI/UX katmaninda iyilestirme yapildi.

## Week-10 Teslimleri

### 1) Smart Upload

- Data sekmesinde tek buton:
  - PDF adindan sirket/ceyrek/yil otomatik algilama
  - otomatik ingest + index_v2
- Manual teknik adimlar `Advanced Mode` altina alindi.

### 2) Company Health Dashboard

- KPI kartlari:
  - Net kar (+/- QoQ)
  - Net marj (renk kodlu)
  - FAVOK marji
  - Revenue growth
  - Store growth
- Config bazli saglik etiketi:
  - `GREEN / YELLOW / RED`
  - esikler `config.yaml -> health`

### 3) Financial Summary Block

- `src/ratio_engine.py`:
  - extracted metric + trend verisinden 5 maddelik executive summary
  - citation odakli grounded metin

### 4) Change Detection Panel

- Son 2 ceyrek karsilastirma:
  - iyilesenler
  - kotulesenler
  - yatay kalanlar

### 5) Clean UI

- Arama modu gibi muhendislik secenekleri varsayilan gizli.
- `Advanced Mode` expander ile aciliyor.

## Week-9 Teslimleri

### 1) Multi-Company Support

- `src/ingest.py`, `src/chunking.py`, `src/retrieve.py` guncellendi.
- Metadata alanlari:
  - `company`, `quarter`, `year` (+ mevcut alanlar)
- Retrieval company filtresi:
  - CLI: `--company`
  - API `/ask`: `"company":"BIM"`
  - company belirtilmezse tum sirketler.

### 2) Financial Ratio Engine

- Yeni modul: `src/ratio_engine.py`
- Hesaplanan oranlar:
  - `net_margin = net_kar / satis_gelirleri`
  - `favok_margin = favok / satis_gelirleri`
  - `revenue_growth_qoq`
  - `store_growth_qoq`

### 3) Cross-Company Comparison Mode

- Query tetikleyicileri:
  - `karsilastir`, `hangisi daha iyi`, coklu sirket adi
- Cikti:
  - karsilastirma tablosu
  - best performer
  - kanit kayitlari

### 4) CSV Export

- API endpoint:
  - `GET /export?type=trend&company=BIM`
  - `GET /export?type=ratio&company=BIM`
- UI:
  - trend tablosu CSV indirme
  - ratio tablosu CSV indirme

### 5) Dashboard Panel

- `app/ui.py` icine yeni `Dashboard` sekmesi eklendi.
- Icerik:
  - Company selector
  - KPI cards
  - Net kar / net marj trend chartlari

## Week-8 Teslimleri

### 1) Architecture & Flow Diagramlari

- `docs/diagrams/architecture.md`
- `docs/diagrams/retrieval_flow.md`
- Mermaid ile:
  - sistem mimarisi
  - retrieval pipeline karsilastirmasi (v1/v3/v5/v6)
  - trend mode (Q1/Q2/Q3 -> pandas -> chart)

### 2) Trade-off Raporu

- `docs/reports/tradeoffs.md`
- `latency_benchmark.json` degerleri ile:
  - `v3`: avg 34.31ms, p95 37.81ms, hit@1 0.80
  - `v5`: avg 31.12ms, p95 37.03ms, hit@1 0.55
  - `v6`: avg 1197.38ms, p95 1178.22ms, hit@1 0.55
- Uretim default karari: v3

### 3) Error Analysis Raporu

- `docs/reports/error_analysis.md`
- `error_analysis.jsonl` uzerinden:
  - failure mode ozetleri
  - ornek vakalar (expected vs retrieved)
  - mitigation fikirleri

### 4) README Quickstart + Demo Script

- README ust bolumu:
  - 60 saniyede kurulum/ingest/index_v2/UI
  - `/ask` icin curl ornegi
  - screenshot placeholder
  - 5 soruluk demo script

### 5) Career Kit

- `docs/career/cv_bullets.md`
- `docs/career/interview_story.md` (STAR format)

### 6) Repo Hijyeni

- `LICENSE` (MIT)
- `CODEOWNERS`
- `.env.example`

## Week-7 Teslimleri

### 1) Config Sistemi

- Yeni dosya: `config.yaml`
- Yeni modul: `src/config.py`
- Kapsam:
  - path ayarlari
  - chroma ve collection isimleri
  - chunk parametreleri
  - retrieval parametreleri
  - model isimleri
  - eval/benchmark dosya yollari
- CLI ve Streamlit varsayilanlari config'ten okunacak sekilde guncellendi.

### 2) FastAPI Servisi

- Yeni dosya: `app/api.py`
- Calistirma:
  - `uvicorn app.api:app --host 0.0.0.0 --port 8000`
- Endpointler:
  - `GET /health`
  - `GET /stats`
  - `POST /ingest`
  - `POST /index`
  - `POST /ask`
- `/ask` hem tek-soru (single) hem trend modunu destekler.
- Grounded davranis korunur (`found=false` + net mesaj).

### 3) Testler

- Yeni testler:
  - `tests/test_query_parser.py`
  - `tests/test_metrics_extractor.py`
  - `tests/test_retriever_smoke.py`
  - `tests/test_api.py`
- Yeni dosya:
  - `requirements-dev.txt` (`pytest`, `httpx`)

### 4) Latency Benchmark

- Yeni modul: `src/latency_benchmark.py`
- Yeni CLI:
  - `python -m src.cli latency_bench`
- Retrievers:
  - `v3`, `v5`, `v6`
- Cikti:
  - `data/processed/latency_benchmark.json`

### 5) Docker

- Yeni dosyalar:
  - `Dockerfile`
  - `.dockerignore`
- Taban imaj:
  - `python:3.11-slim`
- Default command:
  - API servisi (`uvicorn app.api:app ...`)

### 6) Dokumantasyon

- `README.md` Week-7 bolumu eklendi:
  - config kullanimi
  - API calistirma
  - curl ornekleri
  - test calistirma
  - latency benchmark
  - docker komutlari

## Ek Teknik Guncellemeler

- `requirements.txt` eklentileri:
  - `PyYAML`
  - `fastapi`
  - `uvicorn`
  - `rank-bm25` zaten week6 ile aktif

## Son Dogrulama

- `python -m pytest -q` -> `27 passed, 1 skipped`.
- Yeni testler:
  - `tests/test_commentary.py`
  - `tests/test_config_dotenv.py`
- `python -m ragfin.demo --prepare-only` calistirildi.
  - demo workspace hazirlandi: `data/demo/`
  - sample fixture indexleme tamamlandi (`9 pages`, `9 chunks`).
- `python -m ragfin.doctor --config data/demo/config.demo.yaml` calistirildi.
  - tum kritik kontroller `PASS`.
- `python -m src.cli metrics_report --gold-file data/demo_bundle/gold_questions_demo.jsonl --multi-company-gold-file data/demo_bundle/gold_questions_demo_multicompany.jsonl --out data/demo/processed/eval_metrics_summary.json` calistirildi.
- `python scripts/check_demo_metrics_smoke.py data/demo/processed/eval_metrics_summary.json` calistirildi.
  - demo coverage after:
    - `BIM: 0.8333`
    - `MIGROS: 0.6667`
    - `SOK: 0.6667`
- `python -m src.cli metrics_report` calistirildi.
  - Multi-company extraction accuracy (before -> after):
    - `BIM: 1.0000 -> 1.0000 (delta +0.0000)`
    - `MIGROS: 0.7333 -> 0.8000 (delta +0.0667)`
    - `SOK: 0.8000 -> 0.8667 (delta +0.0667)`
  - Ortalama after accuracy:
    - `(1.0000 + 0.8000 + 0.8667) / 3 = 0.8889` (hedef `>= 0.85` saglandi)
- `python -m src.cli benchmark_week6` calistirildi.
- `python -m src.cli error_report` calistirildi.
- `python -m src.cli latency_bench` calistirildi.
- `python -m src.cli -h` icinde yeni komutlar goruldu.
- `streamlit run app/ui.py` smoke test (HTTP 200) gecildi.
- `app.api` /health endpointi test edildi.
- `compileall` ile derleme kontrolu yapildi.
- `python -m compileall app/ui.py app/ui_components.py` calistirildi (UI patch syntax check).

## Ortam Notu

- `chromadb` Python 3.14 ile uyumsuz.
- Onerilen calisma surumu: Python `3.9 - 3.12`.
- Proje dogrulamasi `.venv39` ile yapildi.
