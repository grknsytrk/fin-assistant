# Faz 5 — Snapshot cache TTL'ini yapılandırılabilir yap

**Öncelik:** P1
**Bağımlılık:** yok
**Commit:** `chore(config): make snapshot cache ttl configurable`

---

## Bağlam

`app/financial_snapshot_store.py` içinde `_SNAPSHOT_MEMORY_CACHE_TTL_SECONDS = 1800.0` sabit.
Sorunlar:

1. Çok süreçli kurulumda (ikinci uvicorn worker, ayrı cron süreci, elle SQL yazımı) invalidation
   diğer süreçlere ulaşmıyor; veri **30 dakikaya kadar** bayat görünebiliyor.
2. Bunu düşürmek için env yok.

`_invalidate_memory_cache` sadece aynı süreçte çalışıyor; bu tasarım değişmiyor. Yapılan iş:
TTL'i env'e taşımak + riski kod içinde belgelemek.

**Kapsam dışı:** Cross-process invalidation mekanizması, Redis pub/sub, DB polling.
Bunları bu fazda **yapma**.

Dosyalar:

- `app/financial_snapshot_store.py`
- `tests/test_financial_snapshot_store.py`
- `README.md`

---

## Adım 5.1 — Env okuma yardımcısı

Dosya: `app/financial_snapshot_store.py`

`LOGGER = logging.getLogger(__name__)` satırının hemen altına, sabitlerden önce ekle:

```python
def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        LOGGER.warning("invalid %s=%r; using default %s", name, raw, default)
        return default
```

---

## Adım 5.2 — Sabiti değiştir

Mevcut (satır 31):

```python
_SNAPSHOT_MEMORY_CACHE_TTL_SECONDS = 1800.0
```

Yeni:

```python
# Upper bound only. Writes invalidate this cache inside the same process; a
# separate writer process (another uvicorn worker, a cron job, a manual SQL
# update) is NOT invalidated and can stay stale until this TTL expires. Keep
# this low (e.g. 60) when multiple processes write to the same database.
_SNAPSHOT_MEMORY_CACHE_TTL_SECONDS = max(
    1.0,
    _env_float("RAGFIN_FINANCIAL_SNAPSHOT_MEMORY_CACHE_TTL_SECONDS", 1800.0),
)
```

`LOGGER` satır 19'da tanımlı; `_env_float` ondan sonra geldiği için sorun yok.

---

## Adım 5.3 — Test

Dosya: `tests/test_financial_snapshot_store.py`

Mevcut `test_financial_snapshot_memory_cache_lives_1800_seconds_and_expires` testine **dokunma**;
varsayılanın 1800 olduğunu doğrulamaya devam eder.

Dosyanın sonuna ekle:

```python
def test_env_float_parses_snapshot_cache_ttl_override(monkeypatch) -> None:
    monkeypatch.setenv("RAGFIN_FINANCIAL_SNAPSHOT_CACHE_TEST", "90")
    assert store._env_float("RAGFIN_FINANCIAL_SNAPSHOT_CACHE_TEST", 1800.0) == 90.0

    monkeypatch.setenv("RAGFIN_FINANCIAL_SNAPSHOT_CACHE_TEST", "not-a-number")
    assert store._env_float("RAGFIN_FINANCIAL_SNAPSHOT_CACHE_TEST", 1800.0) == 1800.0
```

> Not: Test dosyasındaki import alias'ı kontrol et. `from app import financial_snapshot_store as store`
> şeklinde değilse mevcut dosyadaki alias'ı kullan; yeni import stili icat etme.

---

## Adım 5.4 — README

Dosya: `README.md`, "Önemli değişkenler" listesi (satır ~149 civarı).

Şu satırı ekle:

```markdown
- `RAGFIN_FINANCIAL_SNAPSHOT_MEMORY_CACHE_TTL_SECONDS` (varsayılan 1800; çok süreçli yazma varsa düşürün)
```

---

## Doğrulama

```powershell
.external\venv311\Scripts\python.exe -m pytest tests/test_financial_snapshot_store.py -q
```

Beklenen: mevcut testler + yeni test geçer (toplam 14 civarı).

Manuel env kontrolü (opsiyonel):

```powershell
$env:RAGFIN_FINANCIAL_SNAPSHOT_MEMORY_CACHE_TTL_SECONDS="60"
.external\venv311\Scripts\python.exe -c "from app import financial_snapshot_store as s; print(s._SNAPSHOT_MEMORY_CACHE_TTL_SECONDS)"
Remove-Item Env:RAGFIN_FINANCIAL_SNAPSHOT_MEMORY_CACHE_TTL_SECONDS
```

Beklenen: `60.0`.

---

## Definition of Done

- [ ] `_env_float` eklendi ve hatalı değerde uyarı verip varsayılana dönüyor
- [ ] TTL sabiti env'den okunuyor, minimum 1.0
- [ ] Çok süreçli bayatlama riski yorumda belgelendi
- [ ] Yeni test eklendi, mevcut 1800 testi değişmedi
- [ ] README'ye env değişkeni eklendi
- [ ] Hedefli pytest yeşil
- [ ] Tek commit atıldı

## Commit

```
chore(config): make snapshot cache ttl configurable
```

---

## Bu fazı ajana verirken kullan

```
docs/perf-fix/05-faz-5-snapshot-ttl.md dosyasını uygula.
Önce docs/perf-fix/00-kurallar.md oku.
Cross-process invalidation ekleme; sadece TTL'i env'e taşı ve belgele.
Mevcut 1800 saniyelik testi değiştirme.
```
