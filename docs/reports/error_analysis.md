# Error Analysis Report (Week-8)

Kaynak: `data/processed/error_analysis.jsonl`  
Toplam hata kaydi: `50`

## Ozet Istatistik

- Retriever bazinda hata:
  - `v1`: 24
  - `v2`: 14
  - `v6_cross`: 9
  - `v4_bm25`: 2
  - `v3`: 1
- Hata tipi dagilimi (top-1 odakli):
  - **Dogru ceyrek, yanlis sayfa**: 24
  - **Dogru ceyrek, yanlis bolum**: 15
  - **Yanlis ceyrek (top-1)**: 11
- Top-1 sonucunda `(no heading)` gorulme sayisi: 31/50.

## Baskin Failure Mode'lar

1. **Page-level sapma (aynı ceyrekte yanlis sayfa):**  
   Benzer finansal ifadeler nedeniyle ayni rapor icinde yanlis tablo/sayfa one cikabiliyor.
2. **Section-level sapma (heading karisimi):**  
   Beklenen bolum yerine genel/metinsel bolumler top-1 olabiliyor.
3. **Quarter karisimi:**  
   Benzer KPI cümleleri Q1/Q2/Q3 arasi gecis yapabiliyor.
4. **Heading zayifligi:**  
   `(no heading)` chunk'lari section-sinyalini dusurdugu icin precision geriliyor.

## Ornekler (Expected vs Retrieved)

1. **q004** - "2025 ilk 9 ay satış gelirleri ne kadar?" (`v4_bm25`)
   - Expected: `Q3, page 9`
   - Retrieved top-1: `Q3, page 3` (bolum: sirket tanitim)

2. **q010** - "2025 ilk çeyrekte yatırım harcaması tutarı nedir?" (`v1`)
   - Expected: `Q1, page 5` (Finansman Kaynaklari)
   - Retrieved top-1: `Q2, page 10`

3. **q014** - "2025 ikinci çeyrekte net kâr geçen yıla göre artmış mı azalmış mı?" (`v2`)
   - Expected: `Q2, page 10`
   - Retrieved top-1: `Q2, page 6`

4. **q017** - "2025 ilk yarıyılda satışlar geçen yıla göre arttı mı?" (`v2`)
   - Expected: `Q2, page 10`
   - Retrieved top-1: `Q1, page 5`

5. **q015** - "2025 birinci çeyrekte FAVÖK geçen yıla göre nasıl değişmiş?" (`v4_bm25`)
   - Expected: `Q1, page 9`
   - Retrieved top-1: `Q1, page 7`

## Mitigation Fikirleri (Uygulanmadi)

1. Heading tespiti iyilestirme:
   - `section_title` bos kalinca komsu chunk heading propagation.
2. Numeric sorularda tablo agirliklandirma:
   - `block_type=table_like` boostunu soruya gore arttirma.
3. Page-aware reranking:
   - Beklenen finansal tablo sayfalarina yakin sayfalara soft prior verme.
4. Quarter confidence gating:
   - Soru ceyregi netse hard quarter filter.
5. Synonym/query expansion:
   - "yatirim harcamasi/capex", "satislar/hasilat/satis gelirleri" genisletme.

