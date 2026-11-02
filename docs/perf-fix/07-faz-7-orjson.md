# Faz 7 — orjson `None` tutarlılığı ve ölçüm notu

**Öncelik:** P3 (düşük risk)
**Bağımlılık:** yok
**Sonraki faz:** `08-faz-8-shutdown.md` (ikisi de `app/api.py`)
**Commit:** `fix(etag): serialize none content as json null`

---

## Bağlam

### Sorun 1: `None` gövdesi

`ORJSONResponse(content=None)` şu an gövdeyi `b""` döndürüyor. FastAPI'nin standart `JSONResponse`'ı
`b"null"` döndürür. Repoda `None` dönen route yok, yani pratik regresyon yok — ama sessiz bir
sözleşme farkı.

Mevcut test bu davranışı açıkça doğruluyor (`tests/test_etag.py:386-387`), yani değişiklik
**bilinçli** bir karar.

### Sorun 2: yanlış performans iddiası

"orjson 100 ms kazandırıyor" iddiası ölçümle desteklenmiyor. 1.500 satırlık payload ile ölçüm:

| Yol | Süre |
|---|---|
| stdlib `json.dumps` | 4,4 ms |
| `jsonable_encoder` + `json.dumps` | 30,3 ms |
| orjson | 0,47 ms |

Kazanç ~4 ms (sadece serializer) veya ~30 ms (`jsonable_encoder` atlanıyorsa). FastAPI normal
route'larda `jsonable_encoder`'ı hâlâ çalıştırıyor, yani gerçek kazanç ~4 ms mertebesinde.
Bu fazda kod optimize edilmiyor; sadece yanlış iddia kod yorumunda düzeltiliyor.

**Kapsam dışı:** `orjson_default`'a yeni tip ekleme, `jsonable_encoder`'ı bypass etme denemesi.

Dosyalar:

- `app/etag.py`
- `tests/test_etag.py`
- `app/api.py` (sadece yorum)

---

## Adım 7.1 — `render(None)` davranışı

Dosya: `app/etag.py`, satır 127-132.

Mevcut:

```python
    def render(self, content: Any) -> bytes:
        if content is None:
            return b""
        if isinstance(content, (bytes, memoryview)):
            return bytes(content)
        return orjson_dumps(content)
```

Yeni:

```python
    def render(self, content: Any) -> bytes:
        if content is None:
            return b"null"
        if isinstance(content, (bytes, memoryview)):
            return bytes(content)
        return orjson_dumps(content)
```

---

## Adım 7.2 — Testi güncelle

Dosya: `tests/test_etag.py`, satır 386-387.

Mevcut:

```python
    empty_resp = ORJSONResponse(content=None)
    assert empty_resp.body == b""
```

Yeni:

```python
    null_resp = ORJSONResponse(content=None)
    assert null_resp.body == b"null"
```

**Dikkat:** `tests/test_etag.py:406` civarındaki `not_modified.body == b""` assertion'ı 304 içindir;
ona **dokunma**. 304 gövdesiz olmalı.

---

## Adım 7.3 — `app/api.py`'ye ölçüm notu

Dosya: `app/api.py`, satır 107-112.

Mevcut:

```python
app = FastAPI(
    title="RAG-Fin API",
    version="0.10.0",
    lifespan=_lifespan,
    default_response_class=ORJSONResponse,
)
```

Yeni:

```python
# orjson only accelerates the final serialization step; FastAPI still runs
# jsonable_encoder for normal routes. Measured saving on a 1,500-row payload is
# ~4 ms, or ~30 ms on the conditional_json_response path which skips the encoder.
app = FastAPI(
    title="RAG-Fin API",
    version="0.10.0",
    lifespan=_lifespan,
    default_response_class=ORJSONResponse,
)
```

"100 ms" gibi bir iddiayı kod yorumuna veya commit mesajına yazma.

---

## Doğrulama

```powershell
.external\venv311\Scripts\python.exe -m pytest tests/test_etag.py -q
```

Beklenen: tüm testler geçer (13 civarı).

Hızlı manuel kontrol:

```powershell
.external\venv311\Scripts\python.exe -c "from app.etag import ORJSONResponse; print(ORJSONResponse(content=None).body)"
```

Beklenen: `b'null'`.

---

## Definition of Done

- [ ] `ORJSONResponse.render(None)` `b"null"` döndürüyor
- [ ] `tests/test_etag.py` içindeki `None` assertion'ı güncellendi
- [ ] 304 testi (`not_modified.body == b""`) değişmedi
- [ ] `app/api.py`'ye ölçüm yorumu eklendi
- [ ] `tests/test_etag.py` yeşil
- [ ] Tek commit atıldı

> **Ürün kararı:** `b""` davranışı bilinçli olarak isteniyorsa bu fazı atla. O durumda
> `render`'a şu yorumu ekle ve commit atma:
> `# Intentionally empty: no route returns None today. JSONResponse would return b"null".`

## Commit

```
fix(etag): serialize none content as json null
```

---

## Bu fazı ajana verirken kullan

```
docs/perf-fix/07-faz-7-orjson.md dosyasını uygula.
Önce docs/perf-fix/00-kurallar.md oku.
Sadece render(None), ilgili test assertion'ı ve api.py'deki yorum değişecek.
orjson_default'a yeni tip ekleme; 100 ms iddiası yazma.
```
