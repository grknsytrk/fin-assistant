# Veri Kaynaklari

Bu dosya uygulamada hangi verinin nereden alindigini sabitler.

## Fonlar

### Fon listesi / gunluk snapshot

1. `fintables_udf_history`
   - URL: `https://gate.fintables.com/barbar/udf/history?symbol={FUND_CODE}&resolution=D&from={start_unix}&to={end_unix}`
   - Kullanim: gunluk collector, fiyat refresh ve snapshot refresh icin tek external fon veri kaynagi.
   - Kural: snapshot refresh mevcut cache veya `RAGFIN_TARGET_FUND_CODES` ile bilinen fon kodlarini Fintables'tan gunceller.

### Fon fiyat history

1. SQLite
   - Tablo: `fund_prices`
   - Unique anahtar: `fund_code + date + source`
   - Kural: `price <= 0` valid performans noktasi sayilmaz.

2. `fintables_udf_history`
   - URL: `https://gate.fintables.com/barbar/udf/history?symbol={FUND_CODE}&resolution=D&from={start_unix}&to={end_unix}`
   - Format: TradingView/UDF (`s`, `t`, `o`, `h`, `l`, `c`, `v`).
   - Kullanim: gunluk fiyat serisi icin ana external history adapter.
   - DB esleme: `t[i]` tarih, `c[i]` gunluk fiyat/close.
   - Not: `fund_prices` tablosuna yazilabilen Fintables kaynagi budur.

3. External fallback yoktur.
   - Fintables basarisiz olursa istek hata veya unavailable durumuyla doner.
   - Eski cache kaynak adlari UI/API'da `legacy_cache` olarak maskelenir.

### Fon donem ozeti

1. `fintables_yield_summary`
   - URL: `https://gate.fintables.com/barbar/server/yield?code={FUND_CODE}`
   - Format: donem bazli ozet (`1w`, `1m`, `3m`, `6m`, `ytd`, `1y`, `3y`, `5y`, `oldest`).
   - Alanlar: `prev_close_date`, `prev_close`, `high`, `low`.
   - Kullanim: performans kartlari, high/low bilgisi, sanity-check ve ozet metrikler.
   - Kural: gunluk seri kaynagi degildir; `fund_prices` tablosuna toplu fiyat noktasi olarak yazilmaz.

### Kapsam filtresi

Sadece su portfoy yonetimi gruplari hedeflenir:

- Tera
- Pusula
- Atlas
- Bulls
- Vega
- Pardus
- Aktif

Konfigurasyon:

- Env: `RAGFIN_TARGET_FUND_MANAGERS`
- Default: `TERA,PUSULA,ATLAS,BULLS,VEGA,PARDUS,AKTIF`
- Env: `RAGFIN_TARGET_FUND_CODES`
- Kullanim: Fintables snapshot/collector icin cache yokken hedef fon kodlarini virgulle ver.

## Hisseler ve piyasa

### Hisse evreni

1. KAP / index universe
   - Kod: `app.kap_service.get_bist_index_universe`
   - Kullanim: `XUTUM`, `XU100`, `XU030` sembol listesi.

### Hisse fiyatlari

1. Info Yatirim
   - URL: `https://infoyatirim.com/canli-borsa`
   - Kullanim: `XUTUM` ve liste fiyatlari icin birincil kaynak.

2. Yahoo Finance
   - Kullanim: bazi endeks/hisse fallback ve intraday chart verisi.

3. Is Yatirim
   - Kullanim: temel ozet, piyasa degeri ve carpanlar.

### Hisse logolari

Backend market listelerinde KAP logosu eager cozulmez; payload `logo_url = null` donebilir.

Frontend `SymbolLogo` sirasi:

1. Fintables company logo
2. Backend `logo_url` varsa onu dene
3. logo.dev domain logo, `VITE_LOGO_DEV_TOKEN` varsa
4. TradingView logo, `VITE_ENABLE_TRADINGVIEW_LOGO_FALLBACK=true` ise
5. Monogram fallback

KAP logo resolver backendde durur:

- URL: `https://www.kap.org.tr/tr/api/member/logo/{oid}`
- Kullanim: gerekirse ayrik/istege bagli endpoint veya ileride cache job.
- Not: market liste response uretirken KAP'a logo icin tek tek gidilmez.

## Fon logolari

Frontend `SymbolLogo(kind="fund")` Fintables fund manager logolarini kullanir.

Birincil kaynak:

- URL: `https://storage.fintables.com/media/uploads/fund-management-logos/{slug}.png`

Desteklenen hedef portfoy aliaslari:

- `tera_portfoy`
- `pusula_portfoy`
- `atlas_portfoy`
- `bulls_portfoy`
- `vega_portfoy`
- `pardus_portfoy`
- `aktif_portfoy`

Fon adi uzun geldiyse prefix eslestirme kullanilir; ornegin `tera_portfoy_birinci_serbest_fon` yine `tera_portfoy_icon.png` adayini uretir.
