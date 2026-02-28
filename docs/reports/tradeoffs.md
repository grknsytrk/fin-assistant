# Trade-off Report (Week-8)

Bu rapor `data/processed/latency_benchmark.json` verisine dayanir.

## Latency vs Accuracy (v3 / v5 / v6)

| Retriever | avg_ms | p95_ms | hit@1 |
|---|---:|---:|---:|
| v3 | 34.31 | 37.81 | 0.8000 |
| v5 | 31.12 | 37.03 | 0.5500 |
| v6 | 1197.38 | 1178.22 | 0.5500 |

## Yorum

- Bu veri setinde en iyi **accuracy/latency dengesi v3**: `hit@1=0.80` ve dusuk gecikme (`34.31 ms`).
- v5 biraz daha hizli olsa da (`31.12 ms`) dogruluk belirgin dusuyor (`hit@1=0.55`).
- v6 cross-encoder, v3'e gore yaklasik **35x daha yavas** (`1197.38 / 34.31 ~= 34.9`) ve dogruluk artisi getirmiyor (`hit@1=0.55`).

## Neden Boyle Olabilir? (Hipotezler)

1. BM25 birlestirmesinde normalizasyon + `beta` agirligi, bu veri setinde iyi vector sonuclarini geri itiyor olabilir.
2. Cross-encoder, aday set kalitesi zayifsa hatayi duzeltemez; sadece gecikme ekler.
3. Cross-encoder modeli (MS MARCO) finansal/TR domainine tam uyumlu olmayabilir.
4. PDF chunk dagilimi (ozellikle table-like ve heading bozulmalari), rerank asamasinda zayif sinyaller uretiyor olabilir.

## Alinan Urun Karari

- **Default production retriever: v3**
- v5 ve v6 kodda korunur, ancak **research toggle** olarak kullanilir.
- UI/API tarafinda varsayilan secim v3 kalir.

## Benchmark Nasil Tekrar Uretilir?

```powershell
.\.venv39\Scripts\python.exe -m src.cli latency_bench
```

Cikti:
- `data/processed/latency_benchmark.json`

Opsiyonel:

```powershell
.\.venv39\Scripts\python.exe -m src.cli benchmark_week6
.\.venv39\Scripts\python.exe -m src.cli error_report
```

