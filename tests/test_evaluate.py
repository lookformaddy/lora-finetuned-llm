from lora_llm.evaluate import _token_f1, parse_response

GOOD = (
    "Category: Network\n"
    "Priority: P2\n"
    "Action: Reinstall the VPN client and switch to the backup gateway."
)


def test_parse_compliant_response():
    parsed = parse_response(GOOD)
    assert parsed is not None
    assert parsed["category"] == "Network"
    assert parsed["priority"] == "P2"
    assert parsed["action"].startswith("Reinstall the VPN client")


def test_parse_rejects_prose():
    assert parse_response("The team discussed the vpn issue at length today.") is None
    assert parse_response("Category: Network") is None  # incomplete
    assert parse_response("Priority: P2\nCategory: Network\nAction: x") is None  # wrong order


def test_token_f1():
    assert _token_f1("unlock the account", "unlock the account") == 1.0
    assert _token_f1("completely different words", "unlock the account") == 0.0
    partial = _token_f1("unlock the account today", "unlock the account")
    assert 0.0 < partial < 1.0
