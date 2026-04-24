"""Tek bildirim için: yerel KAP önbelleği, canlı KAP attachment-detail ve (varsa) VYK html.

Kullanım (repo kökünden):
  .venv\\Scripts\\python.exe scripts\\compare_kap_vyk_financial_probe.py THYAO

Notlar:
  * Canlı KAP: `notification/attachment-detail` + `kap_fetcher._extract_disclosure_metrics`.
  * VYK: `disclosureDetail?fileType=html` içindeki base64 `tr` metni decode edilerek aynı
    çıkarıcı denenir. Test ağ geçidi (`apigwdev`) üretim indekslerinde boş dönebilir.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src import kap_fetcher as kf  # noqa: E402
from src import kap_vyk_client as vyk  # noqa: E402


def _pct_diff(a: float, b: float) -> str:
    if a == b:
        return "aynı"
    if a == 0:
        return "a=0"
    return f"{abs(a - b) / abs(a) * 100:.4f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", nargs="?", default="THYAO", help="kap_cache/{SYMBOL}.json")
    args = parser.parse_args()
    sym = str(args.symbol or "").strip().upper()
    if not sym:
        print("Sembol boş.", file=sys.stderr)
        return 2

    cache_path = ROOT / "data" / "processed" / "kap_cache" / f"{sym}.json"
    if not cache_path.is_file():
        print(f"Önbellek yok: {cache_path}", file=sys.stderr)
        return 2

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    quarters = payload.get("quarters") or []
    if not quarters:
        print("Önbellekte çeyrek yok.", file=sys.stderr)
        return 2

    q = quarters[0]
    idx = int(q["disclosure_index"])
    period = int(q["period"])
    cached = dict(q.get("metrics") or {})

    app = load_config(ROOT / "config.yaml")
    cfg = app.kap

    kap_detail = kf._fetch_attachment_detail(idx, cfg)
    if not kap_detail:
        print(f"KAP attachment-detail boş: {idx}", file=sys.stderr)
        return 3

    live_metrics, unit = kf._extract_disclosure_metrics(kap_detail, period)
    print(f"Şirket: {sym}  çeyrek: {q.get('quarter')}  bildirim: {idx}  dönem kolonu: {period}")
    print(f"Birim (canlı çıkarım): {unit}")
    keys = [
        "net_kar",
        "satis_gelirleri",
        "brut_kar",
        "favok",
        "ozkaynaklar",
        "toplam_varliklar",
        "finansal_borclar",
        "esas_faaliyet_kari",
    ]
    print(f"{'metrik':<20} {'önbellek':>22} {'canlı_KAP':>22} {'değerlendirme':>14}")
    for k in keys:
        c = cached.get(k)
        live = live_metrics.get(k)
        if c is None and live is None:
            continue
        if c is not None and live is not None and isinstance(c, (int, float)) and isinstance(live, (int, float)):
            note = _pct_diff(float(c), float(live))
        elif c == live:
            note = "aynı"
        else:
            note = "farklı/null"
        print(f"{k:<20} {str(c):>22} {str(live):>22} {note:>14}")

    # VYK (html, base64 tr)
    if not vyk.is_enabled(cfg):
        print("\nVYK: yapılandırma yok (api_key/api_secret/vyk_base_url).")
        return 0

    vh = vyk.get_disclosure_detail(cfg, idx, file_type="html")
    if not vh:
        print(
            f"\nVYK: disclosureDetail(html) bu indeks için boş — "
            f"ortam muhtemelen bu bildirimi içermiyor (ör. apigwdev).",
        )
        return 0

    msgs = vh.get("htmlMessages") or []
    if not msgs or not isinstance(msgs[0], dict):
        print("\nVYK: htmlMessages beklenen formatta değil.")
        return 0

    raw_b64 = msgs[0].get("tr")
    if not raw_b64:
        print("\nVYK: htmlMessages[0].tr yok.")
        return 0

    decoded = base64.b64decode(str(raw_b64)).decode("utf-8", errors="replace")
    fake = {"disclosureBody": [decoded]}
    vyk_metrics, vyk_unit = kf._extract_disclosure_metrics(fake, period)
    rows_k = len(kf._extract_rows_from_disclosure_body(kap_detail["disclosureBody"][0], 0, 1.0))
    rows_v = len(kf._extract_rows_from_disclosure_body(decoded, 0, 1.0))
    print(f"\nVYK decode: birim {vyk_unit}  tablo satırı (regex): KAP={rows_k}  VYK={rows_v}")
    print(f"{'metrik':<20} {'canlı_KAP':>22} {'VYK_html':>22}")
    for k in keys:
        a, b = live_metrics.get(k), vyk_metrics.get(k)
        if a is None and b is None:
            continue
        print(f"{k:<20} {str(a):>22} {str(b):>22}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
