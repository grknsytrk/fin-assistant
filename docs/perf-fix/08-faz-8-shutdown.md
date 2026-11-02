# Faz 8 — Shutdown sırasında client kapatma (OPSİYONEL)

**Öncelik:** P3
**Bağımlılık:** Faz 7 (ikisi de `app/api.py`)
**Commit:** `refactor(api): stop refresh executor before closing clients`

> **Bu faz opsiyonel.** Zaman/risk kısıtı varsa atla ve raporla. Doğruluk hatası değil; kapanış
> sırasında görülebilecek hata logu riski.

---

## Bağlam

`app/api.py` lifespan `finally` bloğu:

```python
    finally:
        await _stop_fund_price_collector()
        close_postgres_pool()
        close_shared_http_clients()
```

`_stop_fund_price_collector()` sadece fon fiyat collector'ını durduruyor. KAP flow refresh
executor'ı (`_KAP_FLOW_REFRESH_EXECUTOR`) ve statement refresh job'ları hâlâ çalışıyor olabilir.
Paylaşılan httpx client'ları onların altından kapatılırsa `RuntimeError: Cannot send a request, as
the client has been closed` logları görülebilir.

**Kapsam dışı:** `_kap_post_json` retry davranışı, thread pool boyutu, yeni job iptal mekanizması.

Dosyalar:

- `app/kap_api_service.py`
- `app/api.py`

---

## Adım 8.1 — Executor shutdown fonksiyonu

Dosya: `app/kap_api_service.py`

`_KAP_FLOW_REFRESH_EXECUTOR` tanımının (satır ~40) altına veya modülün uygun bir yerine ekle:

```python
def shutdown_background_refresh_executor() -> None:
    """Stop accepting new KAP refresh jobs during application shutdown."""
    _KAP_FLOW_REFRESH_EXECUTOR.shutdown(wait=False, cancel_futures=True)
```

`cancel_futures=True` Python 3.9+ gerektirir; proje 3.11 kullanıyor, sorun yok.

---

## Adım 8.2 — Lifespan'e ekle

Dosya: `app/api.py`

Import ekle:

```python
from app.kap_api_service import shutdown_background_refresh_executor
```

> Döngüsel import hatası alırsan import'u `_lifespan` fonksiyonunun içine taşı:
> ```python
> from app.kap_api_service import shutdown_background_refresh_executor
> ```
> Import'u başka bir modüle taşıma; fonksiyon `kap_api_service`'te kalacak.

`finally` bloğunu şu hale getir:

```python
    finally:
        await _stop_fund_price_collector()
        shutdown_background_refresh_executor()
        close_postgres_pool()
        close_shared_http_clients()
```

Sıra önemli: executor, client'lardan **önce** durdurulmalı.

---

## Doğrulama

```powershell
.external\venv311\Scripts\python.exe -m pytest tests/test_http_client_pooling.py -q
```

Beklenen: mevcut lifespan ve client pooling testleri bozulmadan geçer.

Manuel (opsiyonel): uygulamayı başlat, bir KAP flow yenilemesi tetikle, hemen Ctrl+C ile kapat.
Loglarda "client has been closed" hatası görünmemeli.

---

## Definition of Done

- [ ] `shutdown_background_refresh_executor` eklendi
- [ ] Lifespan `finally` bloğunda client kapatmadan önce çağrılıyor
- [ ] Döngüsel import yok (veya fonksiyon içi import ile çözüldü)
- [ ] `tests/test_http_client_pooling.py` yeşil
- [ ] Tek commit atıldı

## Commit

```
refactor(api): stop refresh executor before closing clients
```

---

## Bu fazı ajana verirken kullan

```
docs/perf-fix/08-faz-8-shutdown.md dosyasını uygula.
Önce docs/perf-fix/00-kurallar.md oku.
_kap_post_json davranışına dokunma; sadece executor shutdown sırası ekle.
Döngüsel import olursa import'u fonksiyon içine al.
```
