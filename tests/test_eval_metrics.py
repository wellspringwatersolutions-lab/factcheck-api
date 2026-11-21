from factcheck.eval.metrics import accuracy, macro_f1, map_gold, map_verdict


def test_label_map_and_scores():
    assert map_gold("SUPPORTS") == "Supported"
    assert map_gold("REFUTES") == "Contradicted"
    assert map_gold("NOT ENOUGH INFO") == "Unverified"
    assert map_verdict("Unsupported") == "Unverified"
    assert map_verdict("Hypothesis") == "Unverified"
    golds = ["Supported", "Contradicted", "Unverified"]
    preds = ["Supported", "Contradicted", "Unverified"]
    assert accuracy(golds, preds) == 1
    assert macro_f1(golds, preds) == 1
    mixed = ["Supported", "Unverified", "Unverified"]
    assert accuracy(golds, mixed) < 1
