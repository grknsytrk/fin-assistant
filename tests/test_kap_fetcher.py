from src.kap_fetcher import _pick_metric_value


def test_net_kar_prefers_parent_or_net_profit_rows() -> None:
    rows = [
        {
            "label_norm": "surdurulen faaliyetler donem kari (zarari)",
            "body_index": 1,
            "col_order": 4,
            "value": 34_628_000_000.0,
        },
        {
            "label_norm": "net donem kari veya zarari",
            "body_index": 0,
            "col_order": 4,
            "value": 22_001_000_000.0,
        },
        {
            "label_norm": "ana ortaklik paylari",
            "body_index": 1,
            "col_order": 4,
            "value": 22_001_000_000.0,
        },
        {
            "label_norm": "kontrol gucu olmayan paylar",
            "body_index": 1,
            "col_order": 4,
            "value": 50_000_000_000.0,
        },
    ]

    picked = _pick_metric_value("net_kar", rows, period=4)
    assert picked == 22_001_000_000.0


def test_net_kar_falls_back_when_only_continued_operations_exists() -> None:
    rows = [
        {
            "label_norm": "surdurulen faaliyetler donem kari (zarari)",
            "body_index": 1,
            "col_order": 4,
            "value": 3_500_000_000.0,
        }
    ]

    picked = _pick_metric_value("net_kar", rows, period=3)
    assert picked == 3_500_000_000.0
