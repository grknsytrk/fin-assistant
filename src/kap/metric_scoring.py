"""Metric candidate scoring, canonicalization, and extraction from parsed KAP rows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.kap.html_parser import (
    _extract_rows_from_disclosure_body,
    _normalize,
    _parse_unit_info,
)

def _is_favok_label(label_norm: str) -> bool:
    ln = str(label_norm or "")
    if not ln:
        return False
    # Exclude margin-style rows when present.
    if "favok marj" in ln or "fvaok marj" in ln or "ebitda marj" in ln:
        return False
    if "favok" in ln or "fvaok" in ln or "ebitda" in ln:
        return True

    # Expanded wording variants:
    # "faiz/vergi/amortisman/oncesi kar(kazanc)" and close forms.
    has_finance_tax = "faiz" in ln and "vergi" in ln
    has_depr = "amortisman" in ln or "itfa" in ln
    has_profit = "kar" in ln or "kazanc" in ln
    has_before = "oncesi" in ln or "oncesindeki" in ln
    return has_finance_tax and has_depr and has_profit and has_before


def _is_cash_equivalent_label(label_norm: str) -> bool:
    ln = str(label_norm or "")
    if not ln:
        return False
    if ln in {"nakit ve nakit benzerleri", "nakit ve nakit benzeri varliklar"}:
        return True
    if "nakit ve nakit benzer" not in ln:
        return False
    if any(
        token in ln
        for token in (
            "net artis",
            "net azalis",
            "donem basi",
            "donem sonu",
            "cevrim fark",
            "etkisi",
            "diger ",
        )
    ):
        return False
    return True


def _col_preference(
    period: int,
    income_statement: bool,
    prefer_income_statement_ytd: bool = False,
    comparison_mode: str = "current",
    prefer_consolidated_balance: bool = False,
) -> Tuple[int, ...]:
    if comparison_mode == "comparative":
        if income_statement and int(period) > 1:
            return (5, 7, 4, 6) if prefer_income_statement_ytd else (7, 5, 6, 4)
        if income_statement:
            return (5, 7, 4, 6) if prefer_income_statement_ytd else (7, 5, 6, 4)
        if prefer_consolidated_balance:
            return (9, 7, 5, 8, 6, 4)
        return (5, 7, 4, 6)
    if income_statement and int(period) > 1:
        # Income statement rows usually expose: 4=YTD current, 5=YTD prev, 6=quarter current, 7=quarter prev.
        return (4, 6, 5, 7) if prefer_income_statement_ytd else (6, 4, 7, 5)
    if income_statement:
        return (4, 6, 5, 7)
    if prefer_consolidated_balance:
        # Banka bilanço raporlarında sütun düzeni genelde:
        # 4 = TP cari, 5 = YP cari, 6 = toplam cari,
        # 7 = TP önceki, 8 = YP önceki, 9 = toplam önceki.
        # Banka özetinde toplam sütununu öne alıyoruz; eski 4/5/6/7 düzenleri
        # için sonraki sütunlar geriye dönük fallback olarak kalıyor.
        return (6, 4, 5, 9, 7, 8)
    return (4, 6, 5, 7)


def _score_metric_candidate(
    metric_key: str,
    row: Dict[str, Any],
    period: int,
    prefer_income_statement_ytd: bool = False,
    comparison_mode: str = "current",
    prefer_consolidated_balance: bool = False,
) -> int:
    score = 0
    label_norm = str(row.get("label_norm", ""))
    body_index = int(row.get("body_index", -1))
    col_order = int(row.get("col_order", -1))

    if metric_key in {
        "net_kar",
        "satis_gelirleri",
        "brut_kar",
        "favok",
        "faiz_gelirleri",
        "faiz_giderleri",
        "net_ucret_komisyon_gelirleri",
        "net_faaliyet_kari",
        "esas_faaliyet_kari",
        "amortisman_itfa_gideri",
        "prim_uretimi",
        "alinan_net_primler",
        "teknik_gelirler",
        "teknik_denge",
    } and body_index == 1:
        score += 30
    if metric_key in {"faaliyet_nakit_akisi", "capex"} and body_index == 2:
        score += 30
    if metric_key in {
        "donen_varliklar",
        "duran_varliklar",
        "toplam_varliklar",
        "kisa_vadeli_yukumlulukler",
        "finansal_varliklar_net",
        "krediler",
        "mevduatlar",
        "beklenen_zarar_karsiliklari",
        "finansal_borclar",
        "net_borc",
        "ozkaynaklar",
        "nakit_ve_nakit_benzerleri",
        "finansal_varliklar_sigortacilik",
        "esas_faaliyetlerden_alacaklar",
        "teknik_karsiliklar",
        "esas_faaliyetlerden_borclar",
        "odenmis_sermaye",
        "cikarilmis_sermaye",
    } and body_index == 0:
        score += 30

    preferred_cols = _col_preference(
        period=period,
        income_statement=metric_key in {
            "net_kar",
            "satis_gelirleri",
            "brut_kar",
            "favok",
            "faiz_gelirleri",
            "faiz_giderleri",
            "net_ucret_komisyon_gelirleri",
            "net_faaliyet_kari",
            "esas_faaliyet_kari",
            "amortisman_itfa_gideri",
        },
        prefer_income_statement_ytd=prefer_income_statement_ytd,
        comparison_mode=comparison_mode,
        prefer_consolidated_balance=prefer_consolidated_balance,
    )
    if col_order in preferred_cols:
        score += max(1, 20 - preferred_cols.index(col_order) * 6)

    if metric_key == "net_kar":
        # Prefer attributable/explicit net-period profit rows over generic period-profit
        # lines, which may point to pre-attribution or continuing-operations subtotals.
        if "ana ortaklik paylari" in label_norm:
            if label_norm == "ana ortaklik paylari" and body_index == 1:
                score += 135
            elif "donem kari" in label_norm or "net donem kari" in label_norm:
                score += 125
            else:
                score -= 120
        if label_norm == "net donem kari veya zarari":
            score += 140
        elif "net donem kari veya zarari" in label_norm:
            score += 110
        elif "net donem kari" in label_norm:
            score += 90
        elif label_norm == "donem kari (zarari)":
            score += 35
        elif "donem kari (zarari)" in label_norm and "surdurulen faaliyetler" not in label_norm:
            score += 20
        if "kapsamli gelir" in label_norm:
            score -= 100

        # Consolidated statements often include multiple profit layers.
        # We prefer headline net/parent-profit rows over "continued operations".
        if "surdurulen faaliyetler donem kari" in label_norm:
            score -= 60
        if "kontrol gucu olmayan paylar" in label_norm:
            score -= 80
    elif metric_key == "satis_gelirleri":
        if label_norm == "toplam hasilat" or label_norm == "toplam satis gelirleri":
            score += 140
        elif "toplam hasilat" in label_norm or "toplam satis gelirleri" in label_norm:
            score += 115
        if "hasilat" in label_norm:
            score += 35
        if "satis gelirleri" in label_norm:
            score += 25
        if "finans sektoru faaliyetleri hasilati" in label_norm:
            score -= 40
    elif metric_key == "brut_kar":
        if "brut kar" in label_norm and "ticari faaliyetlerden" not in label_norm:
            score += 35
        elif "brut kar" in label_norm:
            score += 25
    elif metric_key == "favok":
        if _is_favok_label(label_norm):
            score += 40
        elif "faaliyet kari" in label_norm and "amortisman" in label_norm:
            score += 20
    elif metric_key == "faiz_gelirleri":
        if "faiz gelirleri" in label_norm:
            score += 45
        if "faiz giderleri" in label_norm:
            score -= 40
    elif metric_key == "faiz_giderleri":
        if "faiz giderleri" in label_norm:
            score += 45
        if "faiz gelirleri" in label_norm:
            score -= 40
    elif metric_key == "net_ucret_komisyon_gelirleri":
        if "net ucret ve komisyon gelirleri" in label_norm:
            score += 45
        elif "net ucret komisyon gelirleri" in label_norm:
            score += 40
    elif metric_key == "net_faaliyet_kari":
        if "net faaliyet kari (zarari)" in label_norm or "net faaliyet kari" in label_norm:
            score += 45
    elif metric_key == "esas_faaliyet_kari":
        if "esas faaliyet kari" in label_norm:
            score += 40
        elif "finansman geliri (gideri) oncesi faaliyet kari" in label_norm:
            score += 20
    elif metric_key == "amortisman_itfa_gideri":
        if "amortisman ve itfa gideri" in label_norm:
            score += 45
        elif "amortisman" in label_norm:
            score += 20
    elif metric_key == "prim_uretimi":
        if "brut yazilan primler" in label_norm:
            score += 110
        elif "yazilan primler" in label_norm and "reasuror payi dusulmus" in label_norm:
            score -= 30
    elif metric_key == "alinan_net_primler":
        if "yazilan primler" in label_norm and "reasuror payi dusulmus" in label_norm:
            score += 105
    elif metric_key == "teknik_gelirler":
        if "hayat disi teknik gelir" in label_norm:
            score += 110
        elif "teknik gelir" in label_norm and "diger teknik gelirler" not in label_norm:
            score += 40
    elif metric_key == "teknik_denge":
        if "teknik bolum dengesi - hayat disi" in label_norm:
            score += 110
        elif "genel teknik bolum dengesi" in label_norm:
            score += 95
    elif metric_key == "faaliyet_nakit_akisi":
        if "isletme faaliyetlerinden nakit akislari" in label_norm:
            score += 35
        elif "faaliyetlerden elde edilen nakit akis" in label_norm:
            score += 25
    elif metric_key == "capex":
        if "duran varliklarin alimindan kaynaklanan nakit cikislari" in label_norm:
            score += 40
        elif "maddi ve maddi olmayan duran varliklarin alimindan kaynaklanan nakit cikislari" in label_norm:
            score += 35
        elif "nakit cikis" in label_norm and "duran varlik" in label_norm:
            score += 20
    elif metric_key == "donen_varliklar":
        if "toplam donen varliklar" in label_norm:
            score += 80
        elif "donen varliklar" in label_norm:
            score += 20
        if "diger donen varliklar" in label_norm:
            score -= 35
    elif metric_key == "duran_varliklar":
        if "toplam duran varliklar" in label_norm:
            score += 80
        elif "duran varliklar" in label_norm:
            score += 20
        if "maddi duran varliklar" in label_norm or "maddi olmayan duran varliklar" in label_norm:
            score -= 35
    elif metric_key == "toplam_varliklar":
        if "toplam varliklar" in label_norm:
            score += 80
    elif metric_key == "finansal_varliklar_net":
        # KAP banka raporunda hem üst toplam ("Finansal Varlıklar (Net)") hem de
        # alt kalem ("İtfa Edilmiş Maliyeti ile Ölçülen Finansal Varlıklar (Net)")
        # benzer etiketle geliyor. Üst toplama yüksek puan veriyor, alt kalem
        # gerekirse düşük puanla eleniyor.
        if label_norm in {"finansal varliklar (net)", "finansal varliklar net"}:
            score += 100
        elif "itfa edilmis" in label_norm and "finansal varliklar" in label_norm:
            score -= 60
        elif "finansal varliklar (net)" in label_norm or "finansal varliklar net" in label_norm:
            score += 60
        elif "finansal varliklar" in label_norm:
            score += 30
    elif metric_key == "krediler":
        # Bankaların KAP bilanço başlığı genelde "Krediler ve Alacaklar" ya da
        # tek başına "Krediler" olur. Müşteri kredilerinin alt kalemleri (ör.
        # "Bireysel Krediler") kazara üstte tutulmasın diye sade etikete tam
        # eşleşme önceliği veriyoruz.
        if label_norm == "krediler" or label_norm == "krediler ve alacaklar":
            score += 90
        elif label_norm.startswith("krediler "):
            score += 50
        elif "krediler" in label_norm:
            score += 25
    elif metric_key == "mevduatlar":
        # KAP banka bilançosunda genelde tek satır: "MEVDUAT". Çoğul
        # "Mevduatlar" formu da var (TFRS uyumlu raporlarda). Alt kalemler
        # ("Bankalar Mevduatı", "Tasarruf Mevduatı") tek başına seçilmesin
        # diye en yüksek puanı en sade ana etiketlere veriyoruz.
        if label_norm in {"mevduat", "mevduatlar"}:
            score += 95
        elif label_norm in {"musteri mevduati", "toplam mevduat", "toplam mevduatlar"}:
            score += 85
        elif label_norm.endswith("mevduati") and "bankalar" not in label_norm and "merkez" not in label_norm:
            score += 30
        elif "mevduatlar" in label_norm or "mevduat" in label_norm:
            score += 15
    elif metric_key == "beklenen_zarar_karsiliklari":
        if "beklenen zarar karsiliklari" in label_norm:
            score += 80
        elif "beklenen kredi zarar karsiliklari" in label_norm:
            score += 70
    elif metric_key == "nakit_ve_nakit_benzerleri":
        if _is_cash_equivalent_label(label_norm):
            score += 110
        if "diger nakit ve nakit benzeri varliklar" in label_norm:
            score -= 80
    elif metric_key == "finansal_varliklar_sigortacilik":
        if "finansal varliklar ile riski sigortalilara ait finansal yatirimlar" in label_norm:
            score += 110
    elif metric_key == "esas_faaliyetlerden_alacaklar":
        if label_norm == "esas faaliyetlerden alacaklar":
            score += 110
        elif "esas faaliyetlerden alacaklar" in label_norm:
            score += 55
    elif metric_key == "teknik_karsiliklar":
        if "sigortacilik teknik karsiliklari" in label_norm:
            score += 110
        elif "teknik karsilik" in label_norm and "diger teknik karsiliklar" not in label_norm:
            score += 50
    elif metric_key == "esas_faaliyetlerden_borclar":
        if label_norm == "esas faaliyetlerden borclar":
            score += 110
        elif "esas faaliyetlerden borclar" in label_norm:
            score += 55
    elif metric_key == "kisa_vadeli_yukumlulukler":
        if "toplam kisa vadeli yukumlulukler" in label_norm:
            score += 80
        elif "kisa vadeli yukumlulukler" in label_norm:
            score += 25
        if "ticari borc" in label_norm or "diger borc" in label_norm:
            score -= 30
    elif metric_key == "finansal_borclar":
        if "finansal borclar" in label_norm or "toplam finansal yukumlulukler" in label_norm:
            score += 40
        elif "kisa vadeli yukumlulukler" in label_norm or "uzun vadeli yukumlulukler" in label_norm:
            score += 15
    elif metric_key == "net_borc":
        if "net borc" in label_norm:
            score += 40
    elif metric_key == "ozkaynaklar":
        if "ana ortakliga ait ozkaynaklar" in label_norm:
            score += 95
        elif "ozsermaye toplami" in label_norm:
            score += 100
        elif "ana ortakliga ait" in label_norm and "ozkaynaklar" in label_norm:
            score += 85
        elif "toplam ozkaynaklar" in label_norm:
            score += 45
        elif "ozkaynaklar" in label_norm:
            score += 30
        elif "ozsermaye" in label_norm:
            score += 30
        if "kontrol gucu olmayan paylar" in label_norm:
            score -= 40
        if "toplam yukumlulukler ve ozsermaye" in label_norm:
            score -= 90
    elif metric_key == "odenmis_sermaye":
        if label_norm == "odenmis sermaye":
            score += 120
        elif "odenmis sermaye" in label_norm:
            score += 80
        if "sermaye duzeltme farklari" in label_norm:
            score -= 120
    elif metric_key == "cikarilmis_sermaye":
        if label_norm == "cikarilmis sermaye":
            score += 120
        elif "cikarilmis sermaye" in label_norm:
            score += 80
        if "sermaye duzeltme farklari" in label_norm:
            score -= 120

    if label_norm.startswith("toplam") or "ara toplam" in label_norm:
        if metric_key in {"donen_varliklar", "duran_varliklar", "toplam_varliklar", "ozkaynaklar"}:
            score += 20
        else:
            score -= 20
    return score


def _pick_metric_value(
    metric_key: str,
    rows: List[Dict[str, Any]],
    period: int,
    prefer_income_statement_ytd: bool = False,
    comparison_mode: str = "current",
    prefer_consolidated_balance: bool = False,
) -> Optional[float]:
    filtered: List[Dict[str, Any]] = []

    if metric_key == "net_kar":
        # Consolidated KAP statements expose both the total consolidated net
        # profit ("Net Dönem Karı veya Zararı") and the profit attributable to
        # the parent ("Ana Ortaklık Payları").  The latter is the comparable
        # net income used by market data screens.  Some reports repeat the
        # parent-profit row in the comprehensive-income section; preserve the
        # first occurrence for each column because it belongs to the income
        # statement.  Previously the explicit total row won first, which made
        # comparative YTD values wrong for companies such as BIMAS.
        parent_rows = [
            (index, row)
            for index, row in enumerate(rows)
            if "ana ortaklik paylari" in str(row.get("label_norm", ""))
            and "kontrol gucu olmayan paylar" not in str(row.get("label_norm", ""))
        ]
        if parent_rows:
            first_parent_by_column: Dict[int, int] = {}
            for index, row in parent_rows:
                col_order = int(row.get("col_order", -1))
                first_parent_by_column.setdefault(col_order, index)
            filtered = [
                row
                for index, row in parent_rows
                if first_parent_by_column.get(int(row.get("col_order", -1))) == index
            ]
        else:
            explicit_net_rows = [
                row
                for row in rows
                if "net donem kari veya zarari" in str(row.get("label_norm", ""))
            ]
            if explicit_net_rows:
                filtered = explicit_net_rows

    for row in rows:
        label_norm = str(row.get("label_norm", ""))
        if metric_key == "net_kar":
            if filtered:
                continue
            if "kontrol gucu olmayan paylar" in label_norm:
                continue
            is_bare_parent_share = "ana ortaklik paylari" in label_norm and (
                "donem kari" not in label_norm and "net donem kari" not in label_norm
            )
            if is_bare_parent_share and int(row.get("body_index", -1)) != 1:
                continue
            if (
                "ana ortaklik paylari" in label_norm
                or "net donem kari veya zarari" in label_norm
                or "net donem kari" in label_norm
                or "donem kari (zarari)" in label_norm
            ):
                filtered.append(row)
        elif metric_key == "satis_gelirleri":
            if "hasilat" in label_norm or "satis gelirleri" in label_norm:
                filtered.append(row)
        elif metric_key == "brut_kar":
            if "brut kar" in label_norm:
                filtered.append(row)
        elif metric_key == "favok":
            if _is_favok_label(label_norm):
                filtered.append(row)
        elif metric_key == "faiz_gelirleri":
            if "faiz gelirleri" in label_norm and "faiz giderleri" not in label_norm:
                filtered.append(row)
        elif metric_key == "faiz_giderleri":
            if "faiz giderleri" in label_norm:
                filtered.append(row)
        elif metric_key == "net_ucret_komisyon_gelirleri":
            if "net ucret ve komisyon gelirleri" in label_norm or "net ucret komisyon gelirleri" in label_norm:
                filtered.append(row)
        elif metric_key == "net_faaliyet_kari":
            if "net faaliyet kari" in label_norm:
                filtered.append(row)
        elif metric_key == "esas_faaliyet_kari":
            if "esas faaliyet kari" in label_norm or "finansman geliri (gideri) oncesi faaliyet kari" in label_norm:
                filtered.append(row)
        elif metric_key == "amortisman_itfa_gideri":
            if "amortisman ve itfa gideri" in label_norm or "amortisman" in label_norm:
                filtered.append(row)
        elif metric_key == "prim_uretimi":
            if "brut yazilan primler" in label_norm:
                filtered.append(row)
        elif metric_key == "alinan_net_primler":
            if "yazilan primler" in label_norm and "reasuror payi dusulmus" in label_norm:
                filtered.append(row)
        elif metric_key == "teknik_gelirler":
            if "hayat disi teknik gelir" in label_norm:
                filtered.append(row)
        elif metric_key == "teknik_denge":
            if "teknik bolum dengesi - hayat disi" in label_norm or "genel teknik bolum dengesi" in label_norm:
                filtered.append(row)
        elif metric_key == "faaliyet_nakit_akisi":
            if "isletme faaliyetlerinden nakit akis" in label_norm or "faaliyetlerden elde edilen nakit akis" in label_norm:
                filtered.append(row)
        elif metric_key == "capex":
            if "nakit cikis" in label_norm and "duran varlik" in label_norm and "alim" in label_norm:
                filtered.append(row)
        elif metric_key == "nakit_ve_nakit_benzerleri":
            if _is_cash_equivalent_label(label_norm):
                filtered.append(row)
        elif metric_key == "finansal_varliklar_sigortacilik":
            if "finansal varliklar ile riski sigortalilara ait finansal yatirimlar" in label_norm:
                filtered.append(row)
        elif metric_key == "esas_faaliyetlerden_alacaklar":
            if label_norm == "esas faaliyetlerden alacaklar" or "esas faaliyetlerden alacaklar" in label_norm:
                filtered.append(row)
        elif metric_key == "teknik_karsiliklar":
            if "sigortacilik teknik karsiliklari" in label_norm:
                filtered.append(row)
            elif "teknik karsilik" in label_norm and "diger teknik karsiliklar" not in label_norm:
                filtered.append(row)
        elif metric_key == "esas_faaliyetlerden_borclar":
            if label_norm == "esas faaliyetlerden borclar" or "esas faaliyetlerden borclar" in label_norm:
                filtered.append(row)
        elif metric_key == "donen_varliklar":
            if "toplam donen varliklar" in label_norm or label_norm == "donen varliklar":
                filtered.append(row)
        elif metric_key == "duran_varliklar":
            if "toplam duran varliklar" in label_norm or label_norm == "duran varliklar":
                filtered.append(row)
        elif metric_key == "toplam_varliklar":
            if "toplam varliklar" in label_norm:
                filtered.append(row)
        elif metric_key == "finansal_varliklar_net":
            if label_norm in {"finansal varliklar (net)", "finansal varliklar net"}:
                filtered.append(row)
            elif "itfa edilmis" in label_norm and "finansal varliklar" in label_norm:
                # Banka bilançosunda alt kalem ("İtfa Edilmiş Maliyeti ile
                # Ölçülen Finansal Varlıklar"). Hâlâ candidate listesine giriyor
                # ama scoring negatif puan vereceği için zayıf öncelik kalıyor.
                filtered.append(row)
            elif "finansal varliklar (net)" in label_norm or "finansal varliklar net" in label_norm:
                filtered.append(row)
            elif label_norm == "finansal varliklar":
                filtered.append(row)
        elif metric_key == "krediler":
            if label_norm in {"krediler", "krediler ve alacaklar"}:
                filtered.append(row)
            elif label_norm.startswith("krediler ") or "krediler" in label_norm:
                filtered.append(row)
        elif metric_key == "mevduatlar":
            if label_norm in {"mevduat", "mevduatlar", "musteri mevduati", "toplam mevduat", "toplam mevduatlar"}:
                filtered.append(row)
            elif "mevduatlar" in label_norm or "mevduat" in label_norm:
                filtered.append(row)
        elif metric_key == "beklenen_zarar_karsiliklari":
            if "beklenen zarar karsiliklari" in label_norm or "beklenen kredi zarar karsiliklari" in label_norm:
                filtered.append(row)
        elif metric_key == "kisa_vadeli_yukumlulukler":
            if "toplam kisa vadeli yukumlulukler" in label_norm or label_norm == "kisa vadeli yukumlulukler":
                filtered.append(row)
        elif metric_key == "finansal_borclar":
            if "finansal borclar" in label_norm or "toplam finansal yukumlulukler" in label_norm:
                filtered.append(row)
        elif metric_key == "net_borc":
            if "net borc" in label_norm:
                filtered.append(row)
        elif metric_key == "ozkaynaklar":
            if "ana ortakliga ait ozkaynaklar" in label_norm:
                filtered.append(row)
            elif "ozsermaye toplami" in label_norm:
                filtered.append(row)
            elif "toplam ozkaynaklar" in label_norm:
                filtered.append(row)
            elif "ozkaynaklar" in label_norm:
                filtered.append(row)
            elif "ozsermaye" in label_norm and "toplam yukumlulukler ve ozsermaye" not in label_norm:
                filtered.append(row)
        elif metric_key == "odenmis_sermaye":
            if label_norm == "odenmis sermaye" or "odenmis sermaye" in label_norm:
                filtered.append(row)
        elif metric_key == "cikarilmis_sermaye":
            if label_norm == "cikarilmis sermaye" or "cikarilmis sermaye" in label_norm:
                filtered.append(row)

    if not filtered:
        return None

    scored = sorted(
        filtered,
        key=lambda item: (
            _score_metric_candidate(
                metric_key,
                item,
                period,
                prefer_income_statement_ytd=prefer_income_statement_ytd,
                comparison_mode=comparison_mode,
                prefer_consolidated_balance=prefer_consolidated_balance,
            ),
            abs(float(item.get("value", 0.0))),
        ),
        reverse=True,
    )
    return float(scored[0].get("value", 0.0))


def _pick_best_row(
    *,
    rows: List[Dict[str, Any]],
    period: int,
    includes: Tuple[str, ...],
    excludes: Tuple[str, ...] = (),
    body_index: Optional[int] = None,
    comparison_mode: str = "current",
) -> Optional[Dict[str, Any]]:
    preferred_cols = _col_preference(
        period=period,
        income_statement=False,
        comparison_mode=comparison_mode,
    )
    candidates: List[Tuple[int, float, Dict[str, Any]]] = []
    for row in rows:
        if body_index is not None and int(row.get("body_index", -1)) != int(body_index):
            continue
        label_norm = str(row.get("label_norm", ""))
        if not all(token in label_norm for token in includes):
            continue
        if any(token in label_norm for token in excludes):
            continue
        col_order = int(row.get("col_order", -1))
        col_rank = preferred_cols.index(col_order) if col_order in preferred_cols else 99
        value = float(row.get("value", 0.0))
        candidates.append((col_rank, -abs(value), row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _derive_finansal_borclar(
    rows: List[Dict[str, Any]],
    period: int,
    comparison_mode: str = "current",
) -> Optional[float]:
    def _bucket(*, includes: Tuple[str, ...], excludes: Tuple[str, ...] = ()) -> Optional[float]:
        non_related_excludes = excludes + ("iliskili taraf",)
        row = _pick_best_row(
            rows=rows,
            period=period,
            includes=includes,
            excludes=non_related_excludes,
            body_index=0,
            comparison_mode=comparison_mode,
        )
        if row is None:
            row = _pick_best_row(
                rows=rows,
                period=period,
                includes=includes,
                excludes=excludes,
                body_index=0,
                comparison_mode=comparison_mode,
            )
        if row is None:
            return None
        return float(row.get("value", 0.0))

    short_term = _bucket(
        includes=("kisa vadeli borclanmalar",),
        excludes=("uzun vadeli borclanmalarin kisa vadeli kisimlari",),
    )
    long_term_current = _bucket(includes=("uzun vadeli borclanmalarin kisa vadeli kisimlari",))
    long_term = _bucket(
        includes=("uzun vadeli borclanmalar",),
        excludes=("kisa vadeli kisim",),
    )

    pieces = [value for value in (short_term, long_term_current, long_term) if value is not None]
    if pieces:
        return float(sum(pieces))

    direct_total = _bucket(
        includes=("finansal", "borc"),
        excludes=("ticari", "diger borc"),
    )
    if direct_total is not None:
        return float(direct_total)
    return None


def _derive_net_borc(
    rows: List[Dict[str, Any]],
    period: int,
    finansal_borclar: Optional[float],
    comparison_mode: str = "current",
) -> Optional[float]:
    direct_row = _pick_best_row(
        rows=rows,
        period=period,
        includes=("net borc",),
        body_index=0,
        comparison_mode=comparison_mode,
    )
    if direct_row is not None:
        return float(direct_row.get("value", 0.0))
    if finansal_borclar is None:
        return None

    nakit_row = _pick_best_row(
        rows=rows,
        period=period,
        includes=("nakit ve nakit benzer",),
        body_index=0,
        comparison_mode=comparison_mode,
    )
    nakit = float(nakit_row.get("value", 0.0)) if nakit_row else 0.0

    cash_like_finansal_yatirim = _pick_cash_like_financial_investment(
        rows=rows,
        period=period,
        comparison_mode=comparison_mode,
    )

    if nakit == 0.0 and cash_like_finansal_yatirim is None:
        return None
    return float(finansal_borclar) - nakit - float(cash_like_finansal_yatirim or 0.0)


def _pick_cash_like_financial_investment(
    *,
    rows: List[Dict[str, Any]],
    period: int,
    comparison_mode: str = "current",
) -> Optional[float]:
    preferred_cols = _col_preference(
        period=period,
        income_statement=False,
        comparison_mode=comparison_mode,
    )
    primary_col = preferred_cols[0] if preferred_cols else 4
    generic_rows: List[Dict[str, Any]] = []
    explicit_rows: List[Dict[str, Any]] = []

    for row in rows:
        if int(row.get("body_index", -1)) != 0:
            continue
        col_order = int(row.get("col_order", -1))
        if col_order != primary_col:
            continue
        label_norm = str(row.get("label_norm", ""))
        if not label_norm:
            continue
        if any(
            token in label_norm
            for token in (
                "vadeli mevduat",
                "para piyasasi fon",
                "likit",
                "nakit benzeri",
                "gercege uygun deger farki kar/zarara yansitilan finansal varliklar",
            )
        ):
            explicit_rows.append(row)
            continue
        if label_norm == "finansal yatirimlar":
            generic_rows.append(row)

    if explicit_rows:
        best_explicit = max(explicit_rows, key=lambda item: abs(float(item.get("value", 0.0))))
        return float(best_explicit.get("value", 0.0))

    # Some issuers expose separate current/non-current investment totals with the same
    # label. In that case the larger current bucket is usually the liquid bucket used
    # in market net debt views. A single generic row is too ambiguous, so we leave it out.
    if len(generic_rows) >= 2:
        best_generic = max(generic_rows, key=lambda item: abs(float(item.get("value", 0.0))))
        return float(best_generic.get("value", 0.0))

    return None


def _pick_income_row_value(
    *,
    rows: List[Dict[str, Any]],
    period: int,
    comparison_mode: str,
    includes: Tuple[str, ...],
    excludes: Tuple[str, ...] = (),
) -> Optional[float]:
    row = _pick_best_row(
        rows=rows,
        period=period,
        includes=includes,
        excludes=excludes,
        body_index=1,
        comparison_mode=comparison_mode,
    )
    if row is None:
        return None
    return float(row.get("value", 0.0))


def _derive_favok(
    *,
    rows: List[Dict[str, Any]],
    period: int,
    comparison_mode: str,
    explicit_favok: Optional[float],
    esas_faaliyet_kari: Optional[float],
    amortisman_itfa_gideri: Optional[float],
) -> Optional[float]:
    if explicit_favok is not None:
        return explicit_favok
    if esas_faaliyet_kari is None or amortisman_itfa_gideri is None:
        return None

    other_income = _pick_income_row_value(
        rows=rows,
        period=period,
        comparison_mode=comparison_mode,
        includes=("esas faaliyetlerden diger gelir",),
    ) or 0.0
    other_expense = _pick_income_row_value(
        rows=rows,
        period=period,
        comparison_mode=comparison_mode,
        includes=("esas faaliyetlerden diger gider",),
    ) or 0.0
    equity_share = _pick_income_row_value(
        rows=rows,
        period=period,
        comparison_mode=comparison_mode,
        includes=("ozkaynak yontemiyle degerlenen", "kar"),
        excludes=("diger kapsamli", "dagitilmamis"),
    ) or 0.0

    try:
        return (
            float(esas_faaliyet_kari)
            - float(other_income)
            - float(other_expense)
            - float(equity_share)
            + float(amortisman_itfa_gideri)
        )
    except Exception:
        return None


def _parse_disclosure_body_rows(
    detail_payload: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    disclosure_body = detail_payload.get("disclosureBody", [])
    if not isinstance(disclosure_body, list):
        return [], {"raw": "", "multiplier": 1.0, "currency": "TL"}

    all_rows: List[Dict[str, Any]] = []
    unit_info = {"raw": "", "multiplier": 1.0, "currency": "TL"}

    for body_index, body_item in enumerate(disclosure_body):
        body_html = str(body_item or "")
        if not body_html:
            continue
        body_unit = _parse_unit_info(body_html)
        if body_unit.get("raw"):
            unit_info = body_unit
        all_rows.extend(
            _extract_rows_from_disclosure_body(
                body_html=body_html,
                body_index=body_index,
                unit_multiplier=float(unit_info.get("multiplier", 1.0)),
            )
        )
    return all_rows, unit_info


def _extract_disclosure_metrics(
    detail_payload: Dict[str, Any],
    period: int,
    *,
    prefer_income_statement_ytd: bool = False,
    comparison_mode: str = "current",
    is_bank: bool = False,
    is_insurance: bool = False,
    parsed_rows: Optional[List[Dict[str, Any]]] = None,
    parsed_unit_info: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Any]]:
    if parsed_rows is not None and parsed_unit_info is not None:
        all_rows = parsed_rows
        unit_info = parsed_unit_info
    else:
        if not isinstance(detail_payload.get("disclosureBody", []), list):
            return {}, {"raw": "", "multiplier": 1.0, "currency": "TL"}
        all_rows, unit_info = _parse_disclosure_body_rows(detail_payload)

    metrics: Dict[str, Optional[float]] = {
        "net_kar": _pick_metric_value(
            "net_kar",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "satis_gelirleri": _pick_metric_value(
            "satis_gelirleri",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "brut_kar": _pick_metric_value(
            "brut_kar",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "favok": _pick_metric_value(
            "favok",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "faiz_gelirleri": _pick_metric_value(
            "faiz_gelirleri",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "faiz_giderleri": _pick_metric_value(
            "faiz_giderleri",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "net_ucret_komisyon_gelirleri": _pick_metric_value(
            "net_ucret_komisyon_gelirleri",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "net_faaliyet_kari": _pick_metric_value(
            "net_faaliyet_kari",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "esas_faaliyet_kari": _pick_metric_value(
            "esas_faaliyet_kari",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "amortisman_itfa_gideri": _pick_metric_value(
            "amortisman_itfa_gideri",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "prim_uretimi": _pick_metric_value(
            "prim_uretimi",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ) if is_insurance else None,
        "alinan_net_primler": _pick_metric_value(
            "alinan_net_primler",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ) if is_insurance else None,
        "teknik_gelirler": _pick_metric_value(
            "teknik_gelirler",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ) if is_insurance else None,
        "teknik_denge": _pick_metric_value(
            "teknik_denge",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ) if is_insurance else None,
        "faaliyet_nakit_akisi": _pick_metric_value("faaliyet_nakit_akisi", all_rows, period=period, comparison_mode=comparison_mode),
        "capex": _pick_metric_value("capex", all_rows, period=period, comparison_mode=comparison_mode),
        "nakit_ve_nakit_benzerleri": _pick_metric_value("nakit_ve_nakit_benzerleri", all_rows, period=period, comparison_mode=comparison_mode, prefer_consolidated_balance=is_bank),
        "finansal_varliklar_sigortacilik": _pick_metric_value("finansal_varliklar_sigortacilik", all_rows, period=period, comparison_mode=comparison_mode) if is_insurance else None,
        "esas_faaliyetlerden_alacaklar": _pick_metric_value("esas_faaliyetlerden_alacaklar", all_rows, period=period, comparison_mode=comparison_mode) if is_insurance else None,
        "teknik_karsiliklar": _pick_metric_value("teknik_karsiliklar", all_rows, period=period, comparison_mode=comparison_mode) if is_insurance else None,
        "esas_faaliyetlerden_borclar": _pick_metric_value("esas_faaliyetlerden_borclar", all_rows, period=period, comparison_mode=comparison_mode) if is_insurance else None,
        "donen_varliklar": _pick_metric_value("donen_varliklar", all_rows, period=period, comparison_mode=comparison_mode, prefer_consolidated_balance=is_bank),
        "duran_varliklar": _pick_metric_value("duran_varliklar", all_rows, period=period, comparison_mode=comparison_mode, prefer_consolidated_balance=is_bank),
        "toplam_varliklar": _pick_metric_value("toplam_varliklar", all_rows, period=period, comparison_mode=comparison_mode, prefer_consolidated_balance=is_bank),
        "kisa_vadeli_yukumlulukler": _pick_metric_value("kisa_vadeli_yukumlulukler", all_rows, period=period, comparison_mode=comparison_mode, prefer_consolidated_balance=is_bank),
        "finansal_varliklar_net": _pick_metric_value("finansal_varliklar_net", all_rows, period=period, comparison_mode=comparison_mode, prefer_consolidated_balance=is_bank),
        "krediler": _pick_metric_value("krediler", all_rows, period=period, comparison_mode=comparison_mode, prefer_consolidated_balance=is_bank),
        "mevduatlar": _pick_metric_value("mevduatlar", all_rows, period=period, comparison_mode=comparison_mode, prefer_consolidated_balance=is_bank),
        "beklenen_zarar_karsiliklari": _pick_metric_value("beklenen_zarar_karsiliklari", all_rows, period=period, comparison_mode=comparison_mode, prefer_consolidated_balance=is_bank),
        "finansal_borclar": _pick_metric_value("finansal_borclar", all_rows, period=period, comparison_mode=comparison_mode, prefer_consolidated_balance=is_bank),
        "net_borc": _pick_metric_value("net_borc", all_rows, period=period, comparison_mode=comparison_mode, prefer_consolidated_balance=is_bank),
        "ozkaynaklar": _pick_metric_value("ozkaynaklar", all_rows, period=period, comparison_mode=comparison_mode, prefer_consolidated_balance=is_bank),
        "odenmis_sermaye": _pick_metric_value("odenmis_sermaye", all_rows, period=period, comparison_mode=comparison_mode, prefer_consolidated_balance=is_bank),
        "cikarilmis_sermaye": _pick_metric_value("cikarilmis_sermaye", all_rows, period=period, comparison_mode=comparison_mode, prefer_consolidated_balance=is_bank),
    }
    if metrics["finansal_borclar"] is None:
        metrics["finansal_borclar"] = _derive_finansal_borclar(
            all_rows,
            period=period,
            comparison_mode=comparison_mode,
        )
    if metrics["net_borc"] is None:
        metrics["net_borc"] = _derive_net_borc(
            all_rows,
            period=period,
            finansal_borclar=metrics.get("finansal_borclar"),
            comparison_mode=comparison_mode,
        )
    metrics["favok"] = _derive_favok(
        rows=all_rows,
        period=period,
        comparison_mode=comparison_mode,
        explicit_favok=metrics.get("favok"),
        esas_faaliyet_kari=metrics.get("esas_faaliyet_kari"),
        amortisman_itfa_gideri=metrics.get("amortisman_itfa_gideri"),
    )
    if metrics["faaliyet_nakit_akisi"] is not None and metrics["capex"] is not None:
        metrics["serbest_nakit_akisi"] = float(metrics["faaliyet_nakit_akisi"]) + float(metrics["capex"])
    else:
        metrics["serbest_nakit_akisi"] = None
    cash_like = metrics.get("nakit_ve_nakit_benzerleri")
    insurance_assets = metrics.get("finansal_varliklar_sigortacilik")
    if cash_like is not None and insurance_assets is not None:
        metrics["nakit_benzeri_finansal_varliklar"] = float(cash_like) + float(insurance_assets)
    else:
        metrics["nakit_benzeri_finansal_varliklar"] = cash_like if cash_like is not None else insurance_assets
    return metrics, unit_info

