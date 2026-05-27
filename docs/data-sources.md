# Veri Kaynaklari

Bu dosya uygulamada hangi verinin nereden alindigini sabitler.

## Fonlar

### Fon listesi / gunluk snapshot

1. `tefasfon_funds`
   - Paket: `tefasfon>=1.1.0,<2`
   - API: `tefasfon.get_funds(fund_type, start_date, end_date, fund_codes=None)`
   - Kullanim: fon kodu, fon adi, fiyat, portfoy buyuklugu, yatirimci sayisi ve liste/snapshot gorunumu.
   - Kural: TEFAS birincil kaynaktir; en son veri olan gun icin snapshot cache yazilir.

2. `tefasfon_returns`
   - API: `tefasfon.get_returns(fund_type, basis="RB")`
   - Kullanim: snapshot satirlarini risk ve donem getirileriyle zenginlestirir.

### Fon fiyat history

1. SQLite
   - Tablo: `fund_prices`
   - Unique anahtar: `fund_code + date + source`
   - Kural: `price <= 0` valid performans noktasi sayilmaz.

2. `tefasfon_funds`
   - API: `tefasfon.get_funds(...)`
   - Kullanim: gunluk fiyat serisi icin ana external history adapter.
   - DB esleme: `fonKodu` fon kodu, `tarih` tarih, `fiyat` gunluk fiyat.

3. `fintables_udf_history`
   - URL: `https://gate.fintables.com/barbar/udf/history?symbol={FUND_CODE}&resolution=D&from={start_unix}&to={end_unix}`
   - Format: TradingView/UDF (`s`, `t`, `o`, `h`, `l`, `c`, `v`).
   - Kullanim: TEFAS/`tefasfon` basarisiz veya eksikse ikinci kaynak.
   - DB esleme: `t[i]` tarih, `c[i]` gunluk fiyat/close.
   - Not: `fund_prices` tablosuna yazilabilen Fintables kaynagi budur.

4. Source policy
   - `tefasfon_primary_fintables_fallback`
   - TEFAS basarisiz olursa veya hedef fon kodu TEFAS sonucunda yoksa Fintables denenir.
   - Eski cache kaynak adlari UI/API'da `legacy_cache` olarak maskelenir.

### Fon donem ozeti

1. `tefasfon_funds`
   - API: fiyat history uzerinden `prev_close`, `high`, `low` donem ozeti hesaplanir.
   - Kullanim: performans kartlari ve high/low bilgisi icin birincil kaynak.

2. `fintables_yield_summary`
   - URL: `https://gate.fintables.com/barbar/server/yield?code={FUND_CODE}`
   - Format: donem bazli ozet (`1w`, `1m`, `3m`, `6m`, `ytd`, `1y`, `3y`, `5y`, `oldest`).
   - Alanlar: `prev_close_date`, `prev_close`, `high`, `low`.
   - Kullanim: TEFAS donem ozeti uretilemezse fallback.
   - Kural: gunluk seri kaynagi degildir; `fund_prices` tablosuna toplu fiyat noktasi olarak yazilmaz.

### Fon portfoy dagilimi

1. `tefasfon_portfolio`
   - API: `tefasfon.get_portfolio(fund_type, start_date, end_date, fund_codes=None)`
   - Kullanim: fon detayindaki varlik dagilimi donut ve tablo gorunumu.
   - Kural: TEFAS birincil kaynaktir; Fintables allocation endpoint'i tanimli degildir.

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
- Kullanim: collector veya detay fiyat backfill icin cache yokken hedef fon kodlarini virgulle ver.
- Env: `RAGFIN_TEFAS_FUND_TYPES`
- Default: `SEC`
- Gecerli degerler: `SEC,PEN,ETF,RE,VC`
- Env: `RAGFIN_TEFAS_OPEN_ONLY`
- Default: `1`
- Kullanim: `tefasfon` satirindaki `tefasDurum=True` olmayan fonlar snapshot ve collector hedef evrenine alinmaz.
- Env: `RAGFIN_FUNDS_FULL_HISTORY_START_DATE`
- Default: `2000-01-01`
- Kullanim: Fon performans grafigi tarih parametresi olmadan istendiginde kurulus/gecmis baslangici olarak kullanilir; gecmis veri tablosu UI'da yine son 30 satirla sinirlidir.

## Hisseler ve piyasa

### Hisse evreni

1. KAP / index universe
   - Kod: `app.kap_service.get_bist_index_universe`
   - Kullanim: `XUTUM`, `XU100`, `XU030` ve desteklenen BIST sektor endeksi sembol listeleri.

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
