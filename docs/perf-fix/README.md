# RAG-FIN Performans Düzeltmeleri — Dosya İndeksi

Bu klasör, tek dosyalık `docs/perf-fix-plan.md` planının **faz başına ayrılmış** halidir.
Amaç: her fazı ayrı bir ajan oturumuna verebilmek ve ajanın kapsam dışına çıkmasını engellemek.

## Neden bölündü

- Tek MD'de ajan 8 fazı birlikte görünce ya hepsini yarım yapıyor ya da sırayı atlıyor.
- Faz dosyası tek başına verildiğinde ajanın "bağlam"ı net oluyor: tek hedef, tek dosya grubu,
  tek doğrulama komutu, tek commit.
- Her fazın kendi "Definition of Done" listesi var; ajan bitirdiğini kanıtlamak zorunda.

## Dosyalar

| Dosya | Faz | Öncelik | Bağımlılık | Dokunduğu dosyalar |
|---|---|---|---|---|
| `00-kurallar.md` | Ortak kurallar | — | — | — |
| `01-faz-1-virtualizer.md` | Sanal tablo clamp hatası | **P0** | — | `useTableVirtualizer.ts`, `FundsPage.tsx` |
| `02-faz-2-aria.md` | ARIA satır bilgisi | P2 | **01** | `FundsPage.tsx`, `MarketsView.tsx` |
| `03-faz-3-flow-head-cache.md` | Edge cache tutarlılığı | P1 | — | `worker/api-cache.ts` |
| `04-faz-4-tatil-takvimi.md` | BIST tatil takvimi | P2 | — | `marketHours.ts` |
| `05-faz-5-snapshot-ttl.md` | Snapshot TTL env | P1 | — | `financial_snapshot_store.py`, `README.md` |
| `06-faz-6-memory-cache.md` | `_MEMORY_CACHE` sınırı | P1 | — | `fund_service.py` |
| `07-faz-7-orjson.md` | orjson `None` + ölçüm notu | P3 | — | `etag.py`, `api.py` |
| `08-faz-8-shutdown.md` | Shutdown sırası (opsiyonel) | P3 | **07** (ikisi de `api.py`) | `kap_api_service.py`, `api.py` |
| `09-nihai-dogrulama.md` | Nihai doğrulama | — | Hepsi | — |
| `10-commit-plani.md` | Commit sırası ve mesajları | — | Hepsi | — |

## Bağımlılık grafiği

```
01 ──► 02

03 ─┐
04 ─┤
05 ─┼──► 09 ──► 10
06 ─┤
07 ──► 08 (opsiyonel)
```

- `01` bitmeden `02` başlamaz: ikisi de `FundsPage.tsx`'e dokunuyor ve `02`, `startIndex`'e bağlı.
- `07` ve `08` ikisi de `app/api.py`'ye dokunduğu için sıralı.
- `03`, `04`, `05`, `06` birbirinden bağımsız; farklı dosyalar. Paralel ajanlara verilebilir.

## Önerilen çalıştırma partileri

| Parti | Fazlar | Not |
|---|---|---|
| A | `01` | Tek başına. En kritik iş. |
| B | `02` | `01` commit'lendikten sonra. |
| C | `03` + `04` + `05` + `06` | Dört ayrı ajanla paralel çalışabilir. |
| D | `07` → `08` | Sıralı. `08` opsiyonel. |
| E | `09` → `10` | Hepsi bittikten sonra tek ajan. |

## Her fazı ajana verirken kullanılacak şablon

```
Sana tek bir faz dosyası veriyorum: docs/perf-fix/<dosya>.md

1. Önce docs/perf-fix/00-kurallar.md dosyasını oku. Oradaki kurallar bu faz için de geçerli.
2. Sadece bu dosyadaki adımları uygula. Sonraki fazlara geçme, kapsam genişletme.
3. Dosyadaki kod bloklarını birebir uygula; "daha iyi yol" gerekçesiyle sapma.
4. Faz sonunda dosyadaki "Doğrulama" komutlarını çalıştır ve çıktısını raporla.
5. "Definition of Done" listesini tek tek işaretle.
6. Son olarak dosyada yazan commit mesajıyla commit at.

Bir şey uyuşmuyorsa dur ve sor; tahminle ilerleme.
```

## Kural: tek doğruluk kaynağı

İki sürüm var:

- `docs/perf-fix-plan.md` — tüm planı tek dosyada toplayan monolith (kurallar + 8 faz + doğrulama + commit)
- `docs/perf-fix/*.md` — aynı içeriğin faz başına bölünmüş hâli (her fazı ayrı ajana vermek için)

Bir faz dosyası ile monolith çelişirse **monolith (`docs/perf-fix-plan.md`) geçerlidir**;
faz dosyaları ondan türetilmiştir.
