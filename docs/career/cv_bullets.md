# CV Bullets (RAG-Fin Case Study)

- BIST perakende raporlari icin local-first bir RAG sistemi tasarladim (pypdf + sentence-transformers + ChromaDB) ve PDF page-level metadata (`doc_id/quarter/page/chunk_id`) ile izlenebilir retrieval kurdum.
- Retrieval kalitesini metrik odakli iyilestirdim; `hit@1` degeri `v1: 0.1667` seviyesinden `v3: 0.7143` seviyesine cikti (yaklasik 4.3x artis).
- Grounded cevap motoru gelistirdim: kanit yoksa sistem zorunlu olarak "Dokümanda bulunamadı" donuyor ve aranan kaynaklari listeliyor.
- Cross-quarter finans analizi (Q1/Q2/Q3) modunu kurdum; metrik extraction + pandas ile trend tablosu, yuzdesel degisim ve yon analizi urettim.
- Streamlit UI ve FastAPI katmanlarini ayni cekirdek modullerle entegre ederek hem demo arayuz hem de API servis kullanimini sagladim.
- Benchmark ve error-analysis pipeline’lari (v1-v6) olusturdum; cross-encoder yolunun bu veri setinde ~35x daha yavas olup kalite kazanci saglamadigini gosterdim.
- CI-benzeri dayanak icin smoke + unit testleri, Docker paketleme ve config-temelli calisma modelini ekleyerek projeyi uretim-benzeri hale getirdim.

