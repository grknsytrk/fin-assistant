# Faz 4 — BIST tatil takvimi

**Öncelik:** P2
**Bağımlılık:** yok
**Commit:** `fix(ui): skip bist polling on full-day holidays`

---

## Bağlam

`isBistTradingHours` sadece hafta sonu ve saat kontrolü yapıyor. Resmî tatillerde panel 5 s'de bir
poll etmeye devam ediyor.

Bu fonksiyon **iki yerde** kullanılıyor:

1. `frontend/src/components/MarketFlowPanel.tsx` → flow-head poll aralığı (5 s / 60 s)
2. `worker/api-cache.ts` → `cacheControlForListRequest` içinde market-stocks TTL'i

İkinci kullanım için not: tatilde `isBistTradingHours` false dönecek ve market-stocks cache'i
300 s'ye çıkacak. Bu **istenen** davranış. `secondsUntilNextBistSession` tatilleri bilmiyor ama
`MAX_MARKET_CACHE_AGE_SECONDS = 300` üst sınırı sayesinde bayat veri riski yok. Bu davranışı
değiştirme.

**Kapsam dışı:** Yarım günler, backend `_TURKEY_MARKET_FULL_DAY_HOLIDAYS`, edge TTL formülü.

Dosyalar:

- `frontend/src/utils/marketHours.ts`
- `frontend/src/utils/marketHours.test.ts` (yeni)

---

## Adım 4.1 — Saat parçalarına tarih ekle

Dosya: `frontend/src/utils/marketHours.ts`

### Formatter

Mevcut:

```ts
const istanbulPartsFormatter = new Intl.DateTimeFormat('en-US', {
    timeZone: ISTANBUL_TIMEZONE,
    weekday: 'short',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
});
```

Yeni:

```ts
const istanbulPartsFormatter = new Intl.DateTimeFormat('en-US', {
    timeZone: ISTANBUL_TIMEZONE,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    weekday: 'short',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
});
```

### Interface

```ts
export interface IstanbulClockParts {
    year: number;
    month: number;
    day: number;
    weekday: string;
    hour: number;
    minute: number;
    minutesOfDay: number;
    /** YYYY-MM-DD in Europe/Istanbul. */
    isoDate: string;
}
```

### `getIstanbulClockParts`

```ts
export function getIstanbulClockParts(now: Date = new Date()): IstanbulClockParts {
    const parts = istanbulPartsFormatter.formatToParts(now);
    let weekday = 'Mon';
    let hour = 0;
    let minute = 0;
    let year = 1970;
    let month = 1;
    let day = 1;

    for (const part of parts) {
        if (part.type === 'weekday') weekday = part.value;
        else if (part.type === 'year') year = Number(part.value);
        else if (part.type === 'month') month = Number(part.value);
        else if (part.type === 'day') day = Number(part.value);
        else if (part.type === 'hour') hour = Number(part.value) % 24;
        else if (part.type === 'minute') minute = Number(part.value);
    }

    const isoDate = `${String(year).padStart(4, '0')}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;

    return {
        year,
        month,
        day,
        weekday,
        hour,
        minute,
        minutesOfDay: hour * 60 + minute,
        isoDate,
    };
}
```

`getIstanbulClockParts`'ın diğer kullanıcıları (`isGlobalFxTradingHours`, `getAdaptiveRefreshMs`)
ek alanlardan etkilenmez; interface genişlemesi geriye dönük uyumlu.

---

## Adım 4.2 — Tatil listesi

`marketHours.ts` içine, `istanbulPartsFormatter` tanımından sonra ekle:

```ts
/**
 * BIST full-day closures, Europe/Istanbul calendar dates.
 * Source: Borsa İstanbul official holiday calendar.
 * Keep in sync with app/fund_service.py::_TURKEY_MARKET_FULL_DAY_HOLIDAYS.
 * Half-days are intentionally NOT listed: polling a few extra hours is harmless.
 */
const BIST_FULL_DAY_HOLIDAYS: ReadonlySet<string> = new Set([
    // 2026 — mirrors app/fund_service.py:3776-3786
    '2026-01-01',
    '2026-03-20',
    '2026-04-23',
    '2026-05-01',
    '2026-05-19',
    '2026-05-27',
    '2026-05-28',
    '2026-05-29',
    '2026-07-15',
    '2026-08-30',
    '2026-10-29',
    // TODO(2027): add verified BIST holiday dates before 2027-01-01.
]);
```

### Veri doğrulama kuralı — bunu atlama

2026 tarihleri `app/fund_service.py:3776-3786` ile birebir aynı; **değiştirme**.

**2027 tarihlerini uydurma.** Ajanda 2027 için resmî BIST takvimi (`borsaistanbul.com` → Tatil
Günleri) doğrulanabiliyorsa ekle. Doğrulanamıyorsa listeye **ekleme** ve yukarıdaki
`TODO(2027)` yorumunu bırak. Dini bayram tarihlerini tahminle yazmak, tatil gününde yanlış
davranışa yol açar.

---

## Adım 4.3 — `isBistTradingHours` güncelle

Mevcut:

```ts
export function isBistTradingHours(now: Date = new Date()): boolean {
    const { weekday, minutesOfDay } = getIstanbulClockParts(now);
    if (weekday === 'Sat' || weekday === 'Sun') {
        return false;
    }
    const sessionStart = 9 * 60 + 50; // 09:50
    const sessionEnd = 18 * 60 + 20;  // 18:20
    return minutesOfDay >= sessionStart && minutesOfDay <= sessionEnd;
}
```

Yeni:

```ts
export function isBistTradingHours(now: Date = new Date()): boolean {
    const { weekday, minutesOfDay, isoDate } = getIstanbulClockParts(now);
    if (weekday === 'Sat' || weekday === 'Sun') {
        return false;
    }
    if (BIST_FULL_DAY_HOLIDAYS.has(isoDate)) {
        return false;
    }
    const sessionStart = 9 * 60 + 50; // 09:50
    const sessionEnd = 18 * 60 + 20;  // 18:20
    return minutesOfDay >= sessionStart && minutesOfDay <= sessionEnd;
}
```

---

## Adım 4.4 — Testler

Yeni dosya: `frontend/src/utils/marketHours.test.ts`

```ts
import { describe, expect, it } from 'vitest';
import { isBistTradingHours } from './marketHours';

describe('isBistTradingHours', () => {
    it('is open on a regular weekday during the session', () => {
        // Wednesday 2026-11-04 12:00 Istanbul (09:00 UTC)
        expect(isBistTradingHours(new Date('2026-11-04T09:00:00.000Z'))).toBe(true);
    });

    it('is closed on weekends', () => {
        expect(isBistTradingHours(new Date('2026-10-25T11:00:00.000Z'))).toBe(false);
    });

    it('is closed outside session hours', () => {
        expect(isBistTradingHours(new Date('2026-11-04T19:00:00.000Z'))).toBe(false);
    });

    it('is closed on a full-day BIST holiday', () => {
        // 2026-10-29 Republic Day, 12:00 Istanbul
        expect(isBistTradingHours(new Date('2026-10-29T09:00:00.000Z'))).toBe(false);
    });
});
```

---

## Doğrulama

```powershell
cd frontend
npm test -- src/utils/marketHours.test.ts src/components/MarketFlowPanel.test.ts src/workerApiCache.test.ts
npm run build
cd ..
```

`workerApiCache.test.ts` de çalıştırılıyor çünkü Worker aynı fonksiyonu import ediyor; mevcut testler
bozulmamalı.

---

## Definition of Done

- [ ] Formatter'a `year/month/day` eklendi
- [ ] `IstanbulClockParts` `year/month/day/isoDate` içeriyor
- [ ] `getIstanbulClockParts` `isoDate` üretiyor
- [ ] `BIST_FULL_DAY_HOLIDAYS` 2026 tarihleriyle eklendi
- [ ] 2027 için doğrulanmamış tarih **eklenmedi**, TODO yorumu bırakıldı
- [ ] `isBistTradingHours` tatil kontrolü yapıyor
- [ ] Yeni test dosyası eklendi ve geçiyor
- [ ] `MarketFlowPanel` ve `workerApiCache` testleri bozulmadı
- [ ] `npm run build` yeşil
- [ ] Tek commit atıldı

## Commit

```
fix(ui): skip bist polling on full-day holidays
```

---

## Bu fazı ajana verirken kullan

```
docs/perf-fix/04-faz-4-tatil-takvimi.md dosyasını uygula.
Önce docs/perf-fix/00-kurallar.md oku.
2027 tatil tarihi uydurma; doğrulayamıyorsan TODO bırak.
Worker cache TTL formülüne (secondsUntilNextBistSession) dokunma.
```
