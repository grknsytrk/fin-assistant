# Interview Story (STAR) - RAG-Fin

## S - Situation

Finansal faaliyet raporlarindan soru-cevap yapan bir sistemde iki ana sorun vardi:
- dogruluk ve kaynak gosterimi guvenilir degildi,
- demo/urunlesme seviyesi dusuktu (sadece temel pipeline).

## T - Task

Hedefim, projeyi hem teknik olarak olgunlastirmak hem de ise alim surecinde anlatilabilir bir "LLM Systems case study" seviyesine cikarmakti:
- metric-driven retrieval iyilestirme,
- grounded answer zorunlulugu,
- UI + API + test + packaging.

## A - Actions

1. Retrieval evolusyonu:
   - v1 baseline -> v2 heading-aware -> v3 query-aware boost ve otomatik quarter filtre.
   - v5 hybrid (vector+BM25) ve v6 cross-encoder denemelerini benchmark mode ile ekledim.

2. Evaluation discipline:
   - gold soru seti, hit@k/MRR/quarter metrics, detailed error log pipeline’i kurdum.

3. Groundedness:
   - cevap motorunda "evidence-first" kuralini uyguladim; kanit yoksa uydurma yerine "Dokümanda bulunamadı" donusu zorunlu.

4. Financial intelligence:
   - Week-5 trend mode: Q1/Q2/Q3 metrik extraction + pandas karsilastirma + UI chart.

5. Productization:
   - Streamlit demo UX, FastAPI endpointleri, config.yaml, Dockerfile, pytest smoke testleri.

## R - Results

- Retrieval kalite artisi: `hit@1` degeri `v1: 0.1667` -> `v3: 0.7143`.
- Guven katmani: tum cevaplar citation ile geliyor; kanitsiz durumda net "not found" davranisi.
- Benchmark ciktisi: v6 cross-encoder bu veri setinde v3’e gore ~35x daha yavas ve hit@1 kazanimi yok.
- Uretim karari: v3 varsayilan retriever; v5/v6 research toggle olarak korunuyor.

## Tasarim Kararlari (Neden Boylesi?)

- **Neden local-first?** Veri gizliligi ve tekrarlanabilirlik.
- **Neden v3 default?** Accuracy/latency dengesi en iyi.
- **Neden v5/v6 kaldirilmadi?** Arastirma ve ileride domain-adaptasyon deneyleri icin kontrollu alternatif.
- **Neden tek config?** CLI/UI/API arasinda parametre drift’ini azaltmak icin.

## 2 Dakikalik Kisa Anlatim (Interview Pitch)

"Bu projede, dokumandan cevap veren bir RAG asistanini sadece calisir hale getirmedim; metriklerle yonetilen bir retrieval gelistirme surecine cevirdim. v1’den v3’e hit@1’i 0.1667’den 0.7143’e cikardim. Ayrica grounded answer zorunlulugu ekleyerek kanitsiz cevaplari engelledim. Sonrasinda trend mode ile Q1-Q2-Q3 karsilastirmalarini tablo ve grafikle sundum. Son adimda API, Docker, test ve benchmark raporlariyla projeyi portfolio/hiring-ready hale getirdim."

