import pytest

from src.api.deriv_client import DerivClient
from src.api.deriv_v2_auth import (
    build_oauth_authorize_url,
    generate_pkce_pair,
    is_legacy_app_id,
    pick_account,
)
from src.api.trade_executor import TradeExecutor
from src.api.trade_monitor import TradeMonitor
from src.strategy.digit_contracts import BARRIER_TYPES


def test_is_legacy_app_id():
    assert is_legacy_app_id("1089") is True
    assert is_legacy_app_id(" 12345 ") is True
    assert is_legacy_app_id("33R2Z6MTElnIWrId8aH3m") is False
    assert is_legacy_app_id("abc") is False


def test_pick_account_demo_prefers_vrtc():
    accounts = [
        {"account_id": "CR123", "group": "real"},
        {"account_id": "VRTC999", "group": "demo"},
    ]
    picked = pick_account(accounts, mode="demo")
    assert picked is not None
    assert picked["account_id"] == "VRTC999"


def test_pick_account_preferred_id():
    accounts = [
        {"account_id": "VRTC1"},
        {"account_id": "VRTC2"},
    ]
    picked = pick_account(accounts, mode="demo", preferred_id="VRTC2")
    assert picked["account_id"] == "VRTC2"


def test_client_auto_selects_v2_for_alphanumeric_app_id():
    client = DerivClient("33R2Z6MTElnIWrId8aH3m", "tok")
    assert client.api_mode == "v2"
    legacy = DerivClient("1089", "tok")
    assert legacy.api_mode == "legacy"


def test_pkce_and_oauth_url():
    verifier, challenge = generate_pkce_pair()
    assert len(verifier) >= 32
    assert challenge and "=" not in challenge
    url = build_oauth_authorize_url(
        "client123",
        "https://example.com/oauth/callback",
        code_challenge=challenge,
        state="st",
    )
    assert "client_id=client123" in url
    assert "code_challenge=" in url
    assert "code_challenge_method=S256" in url
    assert "response_type=code" in url


def test_parse_history_response():
    data = {
        "msg_type": "history",
        "history": {
            "prices": [100.1, 100.2, 100.3],
            "times": [1000, 1001, 1002],
        },
    }
    ticks = DerivClient.parse_history_response(data, "R_100")
    assert len(ticks) == 3
    assert ticks[0]["quote"] == 100.1
    assert ticks[-1]["epoch"] == 1002
    assert ticks[0]["symbol"] == "R_100"


def test_seed_tick_buffer_dedupes():
    client = DerivClient("app", "token")
    hist = [
        {"symbol": "R_100", "quote": 1.0, "epoch": 10},
        {"symbol": "R_100", "quote": 2.0, "epoch": 11},
    ]
    live = [
        {"symbol": "R_100", "quote": 2.5, "epoch": 11},  # same epoch as hist
        {"symbol": "R_100", "quote": 3.0, "epoch": 12},
    ]
    client.seed_tick_buffer("R_100", hist, prepend=True)
    client.seed_tick_buffer("R_100", live, prepend=False)
    buf = client.get_latest_ticks("R_100", 10)
    assert len(buf) == 3
    # epoch 11 should prefer later write (2.5)
    by_ep = {t["epoch"]: t["quote"] for t in buf}
    assert by_ep[11] == 2.5
    assert by_ep[12] == 3.0


def test_client_init():
    client = DerivClient("test_app", "test_token")
    assert client is not None
    assert client.authorized is False
    assert client.get_balance() is None


def test_proposal_payload_with_barrier():
    # Numeric app id → legacy field "symbol"
    client = DerivClient("1089", "token")
    ex = TradeExecutor(client)
    msg = ex.build_proposal("R_100", "DIGITOVER", 1.0, barrier=6)
    assert msg["contract_type"] == "DIGITOVER"
    assert msg["barrier"] == "6"
    assert msg["amount"] == 1.0
    assert msg["symbol"] == "R_100"
    assert "underlying_symbol" not in msg


def test_proposal_v2_uses_underlying_symbol():
    client = DerivClient("33R2Z6MTElnIWrId8aH3m", "token")
    assert client.api_mode == "v2"
    ex = TradeExecutor(client)
    msg = ex.build_proposal("R_100", "DIGITUNDER", 1.5, barrier=4)
    assert msg["underlying_symbol"] == "R_100"
    assert "symbol" not in msg
    assert msg["barrier"] == "4"
    assert msg["contract_type"] == "DIGITUNDER"


def test_proposal_payload_even_no_barrier():
    client = DerivClient("1089", "token")
    ex = TradeExecutor(client)
    msg = ex.build_proposal("R_100", "DIGITEVEN", 0.5, barrier=3)
    assert "barrier" not in msg
    assert "DIGITOVER" in BARRIER_TYPES


def test_proposal_over_barrier_9_clamped():
    client = DerivClient("1089", "token")
    ex = TradeExecutor(client)
    msg = ex.build_proposal("R_100", "DIGITOVER", 1.0, barrier=9)
    assert msg["barrier"] == "8"


def test_proposal_under_barrier_0_clamped():
    client = DerivClient("1089", "token")
    ex = TradeExecutor(client)
    msg = ex.build_proposal("R_100", "DIGITUNDER", 1.0, barrier=0)
    assert msg["barrier"] == "1"


def test_monitor_closed_detection():
    client = DerivClient("app", "token")
    closed = {}

    def on_close(contract, meta):
        closed["profit"] = contract.get("profit")
        closed["meta"] = meta

    mon = TradeMonitor(client, on_close=on_close)
    mon.open_contracts[99] = {"symbol": "R_100"}
    mon.handle_contract_update(
        {
            "msg_type": "proposal_open_contract",
            "proposal_open_contract": {
                "contract_id": 99,
                "is_sold": 1,
                "profit": 0.85,
                "status": "sold",
            },
        }
    )
    assert closed["profit"] == 0.85
    assert mon.open_count() == 0
    mon.handle_contract_update(
        {
            "msg_type": "proposal_open_contract",
            "proposal_open_contract": {
                "contract_id": 99,
                "is_sold": 1,
                "profit": 0.85,
            },
        }
    )
    assert mon.open_count() == 0
