# Faz 3 — flow/head edge cache tutarlılığı

**Öncelik:** P1
**Bağımlılık:** yok
**Commit:** `fix(edge): align flow-head cache ttl with poll cadence`

---

## Bağlam

Üç ayrı sorun var:

1. **TTL / poll uyumsuzluğu:** Edge TTL 3 s, istemci poll aralığı 5 s. Tek istemci için her istek
   cache'i ıskalıyor → sağlanan fayda **%0**.
2. **`refresh_status` kontrolü yok:** `isCacheableListResponse`, flow-head için boş/hata durumunu
   da cache'lenebilir kabul ediyor.
3. **304 belirsizliği:** Gövdesiz 304 yanıtı cache'lenebilir sayılıyor; Workers Cache'in bunu nasıl
   sakladığı test edilmemiş.

**Neden 10 s TTL güvenli:** KAP flow verisi Worker cron ile **dakikada bir** yenileniyor
(`wrangler.production.toml` → `crons = ["* * * * *", ...]`). Yani `latest_cursor` en fazla 60 s'de
bir değişir. 5 s'lik poll zaten verinin değişme hızından 12 kat hızlı. 10 s'lik cache, veri tazeliği
açısından ölçülebilir kayıp yaratmaz.

**Kapsam dışı:** Backend `kap_flow_store`, frontend poll aralığı, `wrangler.*.toml` ayarları.

Dosyalar:

- `worker/api-cache.ts`
- `frontend/src/workerApiCache.test.ts`

---

## Adım 3.1 — TTL'i poll aralığıyla uyumlu yap

Dosya: `worker/api-cache.ts`, satır 205-207.

Mevcut:

```ts
    if (route === "market-flow-head") {
        return "public, max-age=3, stale-while-revalidate=15, stale-if-error=0";
    }
```

Yeni:

```ts
    if (route === "market-flow-head") {
        // The KAP flow is refreshed by cron once per minute, and the panel
        // polls every 5s. A 10s edge TTL is the smallest value that can serve
        // at least every other poll for a single client.
        return "public, max-age=10, stale-while-revalidate=10, stale-if-error=0";
    }
```

Beklenen etki: tek istemci ~%50, 4+ eşzamanlı istemci ~%87 origin azalması. En kötü durumda yeni
KAP satırı 20 s gecikir; bu, 60 s'lik ingest kadansının altında kalır.

> **Ürün kararı:** "5 s'den fazla gecikme olmasın" deniyorsa alternatif, flow-head'i cache'ten
> tamamen çıkarmaktır (`return "no-store";` + `getCacheableListRoute`'tan kaldırma). Varsayılan
> olarak 10 s uygulanır. Bu alternatifi ancak kullanıcı açıkça isterse yap.

---

## Adım 3.2 — `refresh_status` ve cursor kontrolü

Dosya: `worker/api-cache.ts`, `isCacheableListResponse` içinde.

Mevcut satır 248 civarı:

```ts
    if (route === "funds" && status && status !== "ok") return false;
```

Bunun **hemen altına** ekle:

```ts
    if (route === "market-flow-head") {
        const refreshStatus = String(payload.refresh_status || "").trim().toLowerCase();
        if (["empty", "error", "failed", "unavailable"].includes(refreshStatus)) return false;
        if (!payload.latest_cursor) return false;
    }
```

`stale` bilinçli olarak serbest: son geçerli cursor hâlâ kullanılabilir.

---

## Adım 3.3 — 304 yanıtını olduğu gibi geçir

### Kaldır

`isCacheableListResponse` başındaki şu satırı **sil** (satır 230):

```ts
    if (response.status === 304) return true;
```

### Ekle

`applyListCachePolicy` fonksiyonunun en başına (satır ~260):

```ts
export async function applyListCachePolicy(
    response: Response,
    request: Request,
    route: CacheableListRoute,
    now: Date = new Date(),
): Promise<Response> {
    // A 304 has no body; caching it as a standalone entry is not meaningful and
    // would make the edge serve an empty response without revalidation.
    if (response.status === 304) return response;

    const cacheable = await isCacheableListResponse(response, route);
```

---

## Adım 3.4 — Testler

Dosya: `frontend/src/workerApiCache.test.ts`

### Mevcut assertion'ı güncelle (satır ~192-193)

```ts
        expect(flowResponse.headers.get('Cloudflare-CDN-Cache-Control'))
            .toBe('public, max-age=10, stale-while-revalidate=10, stale-if-error=0');
```

### Yeni testler

```ts
    it('does not cache an empty market flow head', async () => {
        const response = await applyListCachePolicy(
            new Response(JSON.stringify({ latest_cursor: null, refresh_status: 'empty' }), {
                headers: { 'Content-Type': 'application/json' },
            }),
            makeRequest('/api/market/flow/head'),
            'market-flow-head',
        );
        expect(response.headers.get('Cache-Control')).toBe('private, no-store');
        expect(response.headers.get('Cloudflare-CDN-Cache-Control')).toBeNull();
    });

    it('passes 304 responses through without edge cache headers', async () => {
        const response = await applyListCachePolicy(
            new Response(null, { status: 304, headers: { ETag: 'W/"abc"' } }),
            makeRequest('/api/funds'),
            'funds',
        );
        expect(response.status).toBe(304);
        expect(response.headers.get('Cloudflare-CDN-Cache-Control')).toBeNull();
        expect(response.headers.get('Cache-Tag')).toBeNull();
    });
```

---

## Doğrulama

```powershell
cd frontend
npm test -- src/workerApiCache.test.ts
npm run build
cd ..
```

Deploy sonrası manuel (deploy bu fazın kapsamı **değil**, sadece bilgi):

```powershell
curl.exe -sI "https://<worker-host>/api/market/flow/head" | Select-String -Pattern "cache-control|cloudflare-cdn-cache-control|cache-tag"
```

Beklenen: `Cloudflare-CDN-Cache-Control: public, max-age=10, stale-while-revalidate=10, stale-if-error=0`
ve `Cache-Tag: market-flow-head`.

---

## Definition of Done

- [ ] `cacheControlForListRequest` flow-head değeri 10/10/0 oldu
- [ ] `isCacheableListResponse` flow-head için `refresh_status` ve `latest_cursor` kontrol ediyor
- [ ] 304 için `return true` kaldırıldı
- [ ] `applyListCachePolicy` 304'ü değiştirmeden döndürüyor
- [ ] Mevcut test assertion'ı güncellendi
- [ ] İki yeni test eklendi ve geçiyor
- [ ] `npm run build` yeşil
- [ ] Tek commit atıldı

## Commit

```
fix(edge): align flow-head cache ttl with poll cadence
```

---

## Bu fazı ajana verirken kullan

```
docs/perf-fix/03-faz-3-flow-head-cache.md dosyasını uygula.
Önce docs/perf-fix/00-kurallar.md oku.
Sadece worker/api-cache.ts ve frontend/src/workerApiCache.test.ts dosyalarına dokun.
wrangler.toml / wrangler.production.toml dosyalarını değiştirme.
```
