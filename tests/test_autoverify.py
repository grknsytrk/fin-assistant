from src.autoverify import auto_verify_metric


def test_autoverify_pass_for_consistent_candidate() -> None:
    selected = {
        "metric": "net_kar",
        "value": 123_400_000.0,
        "unit": "TL",
        "multiplier": 1_000_000.0,
        "excerpt": "Net kar 123,4 milyon TL olarak gerceklesti.",
        "score": 82.0,
        "reasons": ["year_alignment"],
        "quarter": "Q1",
    }
    candidates = [
        {**selected, "validation_ok": True},
        {
            "metric": "net_kar",
            "value": 122_900_000.0,
            "unit": "TL",
            "multiplier": 1_000_000.0,
            "excerpt": "Net kar 122,9 milyon TL",
            "score": 77.0,
            "validation_ok": True,
        },
    ]
    result = auto_verify_metric("net_kar", selected=selected, candidates=candidates, quarter="Q1")
    assert result["status"] == "PASS"
    assert "alternate_regex_match" in result["checks"]


def test_autoverify_warn_for_candidate_disagreement() -> None:
    selected = {
        "metric": "net_kar",
        "value": 200_000_000.0,
        "unit": "TL",
        "multiplier": 1_000_000.0,
        "excerpt": "Net kar 200 milyon TL",
        "score": 90.0,
        "reasons": [],
        "quarter": "Q2",
    }
    candidates = [
        {**selected, "validation_ok": True},
        {
            "metric": "net_kar",
            "value": 420_000_000.0,
            "unit": "TL",
            "multiplier": 1_000_000.0,
            "excerpt": "Net kar 420 milyon TL",
            "score": 86.0,
            "validation_ok": True,
        },
    ]
    result = auto_verify_metric("net_kar", selected=selected, candidates=candidates, quarter="Q2")
    assert result["status"] == "WARN"
    assert "multiple_strong_candidates_disagree" in result["warnings"]
