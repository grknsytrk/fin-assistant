# Faz 6 — `_MEMORY_CACHE` sınırı

**Öncelik:** P1
**Bağımlılık:** yok
**Commit:** `fix(funds): bound process-local fund cache entries`

---

## Bağlam

`app/fund_service.py` içindeki `_MEMORY_CACHE` düz bir `dict` ve **sınırsız**. Anahtar yapıları:

- `latest-fund-price-points:{db_path}:{end_date}:{code1,code2,...}` → her farklı arama/filtre
  kombinasyonu yeni girdi
- snapshot dosya cache anahtarları

Overlay TTL'i 60 → 900'e çıkarıldığı için girdilerin ortalama ömrü arttı; bellek baskısı da arttı.
Çözüm: FIFO tabanlı üst sınır.

**Kapsam dışı ve kritik:** `_MEMORY_CACHE.clear()` çağrılarının semantiğini **değiştirme**.
Onlar fiyat/snapshot yazımında tüm cache'i geçersiz kılıyor; dokunursan bayat fiyat okuması olur.

Dosyalar:

- `app/fund_service.py`
- `tests/test_fund_service.py`

---

## Adım 6.1 — Sabit ve yardımcı fonksiyon

Dosya: `app/fund_service.py`

Mevcut (satır ~210):

```python
_MEMORY_CACHE: Dict[str, Dict[str, Any]] = {}
```

Bunun **hemen altına** ekle:

```python
try:
    _MEMORY_CACHE_MAX_ENTRIES = max(
        16,
        int(os.getenv("RAGFIN_FUNDS_MEMORY_CACHE_MAX_ENTRIES", "256")),
    )
except ValueError:
    _MEMORY_CACHE_MAX_ENTRIES = 256


def _memory_cache_put(cache_key: str, value: Dict[str, Any]) -> None:
    """Insert into the process-local cache, evicting oldest entries first.

    This is FIFO, not LRU: reads do not refresh insertion order. 256 entries is
    far above the number of distinct fund list/detail views a single process
    serves between deploys, so eviction should stay rare.
    """
    _MEMORY_CACHE.pop(cache_key, None)
    _MEMORY_CACHE[cache_key] = value
    overflow = len(_MEMORY_CACHE) - _MEMORY_CACHE_MAX_ENTRIES
    if overflow <= 0:
        return
    for stale_key in list(_MEMORY_CACHE.keys())[:overflow]:
        _MEMORY_CACHE.pop(stale_key, None)
```

`os`, `Dict`, `Any` zaten import edilmiş; yeni import gerekmiyor.

---

## Adım 6.2 — Yazma noktalarını değiştir

Sadece **iki** yer.

### 1) `read_latest_fund_price_points` (satır ~3155)

Mevcut:

```python
    _MEMORY_CACHE[cache_key] = {"cached_at": time.time(), "points": points}
```

Yeni:

```python
    _memory_cache_put(cache_key, {"cached_at": time.time(), "points": points})
```

### 2) Snapshot dosya cache'i (satır ~6531)

Mevcut:

```python
            if stat:
                _MEMORY_CACHE[cache_key] = {
                    "file_signature": file_signature,
                    "mtime": stat.st_mtime,
                    "payload": payload,
                }
```

Yeni:

```python
            if stat:
                _memory_cache_put(cache_key, {
                    "file_signature": file_signature,
                    "mtime": stat.st_mtime,
                    "payload": payload,
                })
```

Başka `_MEMORY_CACHE[...] = ...` ataması **yok**; varsa `grep` ile doğrula ve aynı helper'a çevir.
Okuma (`_MEMORY_CACHE.get`) ve `clear()` çağrılarına dokunma.

---

## Adım 6.3 — Test

Dosya: `tests/test_fund_service.py`

```python
def test_memory_cache_evicts_oldest_entries(monkeypatch) -> None:
    from app import fund_service

    fund_service.reset_fund_caches_for_tests()
    monkeypatch.setattr(fund_service, "_MEMORY_CACHE_MAX_ENTRIES", 4)

    for index in range(6):
        fund_service._memory_cache_put(f"key-{index}", {"value": index})

    assert len(fund_service._MEMORY_CACHE) == 4
    assert "key-0" not in fund_service._MEMORY_CACHE
    assert "key-1" not in fund_service._MEMORY_CACHE
    assert "key-5" in fund_service._MEMORY_CACHE

    fund_service.reset_fund_caches_for_tests()
```

`reset_fund_caches_for_tests` mevcut (`fund_service.py:434`); yeni fixture yazma.

---

## Adım 6.4 — Opsiyonel: overlay TTL testi

Bu adım **zorunlu değil**. Zaman varsa yap; yoksa atla ve raporla.

Yeni dosya: `tests/test_fund_overlay_cache.py`

Amaç: `read_latest_fund_price_points` TTL içinde ikinci kez DB'ye gitmiyor, TTL sonrası gidiyor.

İskelet:

```python
from __future__ import annotations

from pathlib import Path

from app import fund_service


def test_latest_price_overlay_cache_respects_ttl(tmp_path: Path, monkeypatch) -> None:
    fund_service.reset_fund_caches_for_tests()
    processed_dir = tmp_path
    fund_service.ensure_fund_prices_schema(processed_dir)

    # TODO: insert one fund_prices row using the same helper pattern as the
    # existing tests in tests/test_fund_service.py. Then monkeypatch
    # fund_service.time.time to a fake clock, count calls to
    # fund_service._connect_fund_prices_db, and assert:
    #   call 1 -> DB read
    #   call 2 within 900s -> no DB read
    #   call 3 after 901s -> DB read
    #
    # If wiring this takes more than ~20 lines, skip the test entirely and say
    # so in the final report. Do not invent a new DB fixture.
```

---

## Doğrulama

```powershell
.external\venv311\Scripts\python.exe -m pytest tests/test_fund_service.py -q -k "memory_cache"
```

Beklenen: yeni test geçer.

Regresyon kontrolü (helper'a çevirdiğin iki yol için):

```powershell
.external\venv311\Scripts\python.exe -m pytest tests/test_fund_service.py -q -k "snapshot or overlay or latest"
```

Beklenen: mevcut testler bozulmaz. Çok uzun sürerse `-k` filtresini daralt; tüm dosyayı çalıştırmak
zorunda değilsin.

---

## Definition of Done

- [ ] `_MEMORY_CACHE_MAX_ENTRIES` env'den okunuyor, hatalı değerde 256'ya düşüyor
- [ ] `_memory_cache_put` FIFO eviction yapıyor
- [ ] İki yazma noktası helper'a çevrildi
- [ ] Başka doğrudan atama kalmadı (grep ile doğrulandı)
- [ ] `_MEMORY_CACHE.clear()` çağrıları değişmedi
- [ ] Eviction testi eklendi ve geçiyor
- [ ] Opsiyonel overlay testi yapıldı veya atlandığı raporlandı
- [ ] Hedefli pytest yeşil
- [ ] Tek commit atıldı

## Commit

```
fix(funds): bound process-local fund cache entries
```

---

## Bu fazı ajana verirken kullan

```
docs/perf-fix/06-faz-6-memory-cache.md dosyasını uygula.
Önce docs/perf-fix/00-kurallar.md oku.
_MEMORY_CACHE.clear() çağrılarını KESİNLİKLE değiştirme.
Sadece iki yazma noktasını helper'a çevir; başka refactor yapma.
```
