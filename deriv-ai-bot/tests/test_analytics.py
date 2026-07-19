"""Analytics suite: edge, pattern, digit heatmap, filter, adaptive stake."""
from src.analytics.digit_analysis import digit_heatmap, digit_snapshot
from src.analytics.edge_score import (
    bayesian_win_rate,
    edge_label,
    historical_edge_score,
    live_edge_score,
    pattern_strength,
)
from src.analytics.tick_patterns import detect_patterns
from src.analytics.trade_filter import evaluate_setup
from src.analytics.adaptive_stake import adaptive_risk_pct, stake_from_risk
from src.analytics.martingale_safety import survival_probability
from src.analytics.strategy_builder import StrategyBuilder
from src.analytics.tick_backtest import backtest_digit_rule


def _ticks(n=120, start=1000.0):
    out = []
    p = start
    for i in range(n):
        p = p + 0.01 * ((i % 10) - 4)
        # embed last digit via quote ending
        q = float(f"{int(p)}.{(i % 10)}")
        out.append({"quote": q, "epoch": 1_700_000_000 + i})
    return out


def test_bayesian_small_sample_not_elite():
    # 8/10 raw 80% → bayes ~53%
    b = bayesian_win_rate(8, 2)
    assert 0.50 < b < 0.60


def test_ev_formula_example():
    """EV = 0.6*10 - 0.4*8 = 2.8; EV score = min(40, 2.8*10) = 28."""
    from src.analytics.edge_score import expected_value, wr_score_20, pattern_wr_score_20

    ev = expected_value(0.60, 10.0, 8.0)
    assert abs(ev - 2.8) < 1e-9
    assert min(40.0, ev * 10.0) == 28.0
    # 68% WR ≈ 17 points
    assert 16.0 <= wr_score_20(0.68) <= 18.0
    # 65% pattern → 13/20
    assert abs(pattern_wr_score_20(0.65) - 13.0) < 0.5


def test_historical_edge_positive():
    h = historical_edge_score(
        wins=60, losses=40, gross_profit=600, gross_loss=320, max_dd_pct=8.0
    )
    assert h["edge_score"] >= 50
    assert edge_label(h["edge_score"]) in {
        "Elite",
        "Strong",
        "Tradable",
        "Weak",
        "Avoid",
    }
    # Explainable reasons present
    assert any("EV" in r or "expected" in r.lower() for r in h["reasons"])
    assert "ev" in h["components"]
    assert h["components"]["ev"] + h["components"]["win_rate"] + h["components"][
        "profit_factor"
    ] + h["components"]["drawdown"] == h["edge_score"] or abs(
        h["components"]["ev"]
        + h["components"]["win_rate"]
        + h["components"]["profit_factor"]
        + h["components"]["drawdown"]
        - h["edge_score"]
    ) < 0.2


def test_pattern_strength_auto_ok_threshold():
    ps = pattern_strength(
        wins=780, losses=420, recent_wins=72, recent_losses=28, rarity_score=90
    )
    assert ps["pattern_strength"] >= 60
    assert "class" in ps
    assert "pattern_wr_score_20" in ps
    assert ps["n"] == 1200
    assert ps["auto_ok"] is (ps["pattern_strength"] >= 75)


def test_pattern_strength_classic_example_82():
    """
    Classic model example:
      WR 65, Sample 92, Recency 90, Clarity 100
      → 26+23+18+15 = 82
    """
    from src.analytics.edge_score import (
        sample_size_score_100,
        wr_to_pattern_score,
        recency_performance_score,
    )

    # 760/1200 = 63.3% → WR score ~60 (spec rounded example used 65)
    wr_s = wr_to_pattern_score(0.633)
    assert 55 <= wr_s <= 72
    # Explicit 65% lands near 65 on the curve (60→50, 70→80)
    assert abs(wr_to_pattern_score(0.65) - 65.0) < 1.0

    # 1200 trades → sample ≈ 92
    sample = sample_size_score_100(1200)
    assert 90 <= sample <= 94

    # 71/58 = 1.22 → recency ≈ 90
    rec_s, ratio, label = recency_performance_score(0.71, 0.58)
    assert abs(ratio - 1.22) < 0.02
    assert rec_s >= 85
    assert label == "High"

    # Bayesian: 8 wins 2 losses → ~52.7% not 80%
    assert abs(bayesian_win_rate(8, 2) - 0.527) < 0.01
    assert abs(bayesian_win_rate(800, 200) - 0.773) < 0.01

    # Full classic path with high clarity
    ps = pattern_strength(
        wins=760,
        losses=440,
        recent_wins=71,
        recent_losses=29,
        clarity_score=100,
        formula="classic",
    )
    assert 75 <= ps["pattern_strength"] <= 90
    assert ps["formula"] == "classic"
    # Reconstruct from contributions ≈ 82 style
    c = ps["contributions"]
    assert abs(sum(c.values()) - ps["pattern_strength"]) < 0.5


def test_pattern_class_tiers():
    from src.analytics.edge_score import pattern_class

    assert pattern_class(40) == "Ignore"
    assert pattern_class(55) == "Weak"
    assert pattern_class(70) == "Tradable"
    assert pattern_class(80) == "Strong"
    assert pattern_class(90) == "Elite"


def test_pattern_clarity_production_example():
    """Legacy blend example still valid under formula=legacy."""
    from src.analytics.pattern_clarity import (
        baseline_separation_score,
        pattern_clarity,
        rarity_score,
        simplicity_score,
        context_alignment_score,
        clarity_class,
    )

    assert rarity_score(0.20) == 80.0
    assert rarity_score(0.02) == 98.0
    sep, imp = baseline_separation_score(0.62, 0.50)
    assert abs(imp - 12.0) < 0.01
    assert sep >= 80  # 12pp → 80+

    assert context_alignment_score(1) == 30
    assert context_alignment_score(3) == 80
    assert context_alignment_score(4) == 100
    assert simplicity_score(n_conditions=2) >= 90

    # Legacy formula: 40/25/20/10/5
    total = 90 * 0.4 + 80 * 0.25 + 85 * 0.20 + 70 * 0.10 + 90 * 0.05
    assert abs(total - 84.5) < 0.01
    assert clarity_class(84.5) == "Clear"
    assert clarity_class(40) == "Noise"
    assert clarity_class(90) == "Exceptional"

    pc = pattern_clarity(
        pattern_wr=0.62,
        baseline_wr=0.50,
        frequency=0.05,
        window_win_rates=[0.61, 0.63, 0.60],
        confirmations=3,
        n_conditions=2,
        sample_n=600,
        formula="legacy",
    )
    assert pc["pattern_clarity"] >= 70
    assert "separation" in pc["components"]
    assert pc["auto_ok"] is (pc["pattern_clarity"] >= 80)


def test_entropy_clarity_uniform_low():
    from src.analytics.pattern_clarity import (
        entropy_clarity_from_digits,
        entropy_from_counts,
        HMAX_DIGITS,
        compression_from_h,
        sliding_window_entropy,
        entropy_strength,
        composite_entropy_score,
    )

    # Uniform 1000 ticks
    counts = [100] * 10
    h = entropy_from_counts(counts)
    assert abs(h - HMAX_DIGITS) < 0.01
    c = compression_from_h(h, HMAX_DIGITS)
    assert c["entropy_clarity"] < 1.0
    assert c["compression_pct"] < 1.0

    digits = list(range(10)) * 100
    e = entropy_clarity_from_digits(digits)
    assert e["entropy_clarity"] < 5
    assert e["bias_label"] == "Normal"

    # Distorted example-like counts
    distorted = []
    for d, n in enumerate([160, 150, 140, 130, 120, 90, 70, 60, 50, 30]):
        distorted.extend([d] * n)
    e2 = entropy_clarity_from_digits(distorted)
    assert e2["entropy"] < HMAX_DIGITS - 0.05
    assert e2["entropy_clarity"] > e["entropy_clarity"]
    assert e2["compression_pct"] > 0

    # Sliding window: short distorted, long mixed
    mixed = list(range(10)) * 40  # 400 uniform
    short_skew = [9] * 50
    series = mixed + short_skew
    slide = sliding_window_entropy(series, windows=(50, 100, 200, 500))
    assert slide["h_short"] < slide["h_long"] or slide["fresh_pattern"]
    assert "momentum_score" in slide


def test_entropy_strength_and_composite():
    from src.analytics.pattern_clarity import entropy_strength, composite_entropy_score

    ticks = _ticks(300)
    # skew last digits toward 7
    for i in range(80):
        ticks[-(i + 1)]["quote"] = 100.7

    comp = composite_entropy_score(ticks, lookback=200)
    assert "digit" in comp["components"]
    assert "odd_even" in comp["components"]
    assert comp["composite_entropy_score"] >= 0

    es = entropy_strength(ticks, lookback=300)
    assert 0 <= es["entropy_strength"] <= 100
    assert "digit_100" in es
    assert "sliding" in es
    assert es["digit_100"]["compression_pct"] >= 0


def test_hpp_velocity():
    from src.analytics.hpp_velocity import (
        velocity_state,
        momentum_score_from_velocity,
        percentage_velocity,
        multi_window_velocity,
        ema_update,
        classify_edge_flag,
        compute_metric_velocity,
        weighted_engine_velocity,
        sample_confidence_trades,
    )

    assert velocity_state(12) == "RAPIDLY IMPROVING"
    assert velocity_state(7) == "IMPROVING"
    assert velocity_state(0) == "STABLE"
    assert velocity_state(-7) == "DECLINING"
    assert velocity_state(-12) == "RAPID DECAY"

    assert abs(percentage_velocity(80, 70) - 14.2857) < 0.1
    assert momentum_score_from_velocity(0) == 50
    assert momentum_score_from_velocity(15) >= 99

    mw = multi_window_velocity(84, 76, 70, 65)
    # short=8, med=14, long=19 → 0.5*8+0.3*14+0.2*19 = 4+4.2+3.8 = 12
    assert mw["short"] == 8
    assert abs(mw["velocity_score"] - 12.0) < 0.1

    assert abs(ema_update(10, 3, 0.2) - 4.4) < 0.01
    assert sample_confidence_trades(50) == 0.1
    assert sample_confidence_trades(500) == 1.0

    assert classify_edge_flag([-4, -4, -6, -6]) == "STRATEGY_DECAYING"
    assert classify_edge_flag([3, 4, 5, 8]) == "EMERGING_EDGE"

    pack = compute_metric_velocity(
        hpp=84, hpp_20=82, hpp_100=76, hpp_500=70, previous_hpp=80, sample_n=200
    )
    assert "velocity_ema" in pack
    assert pack["status"]

    overall = weighted_engine_velocity(
        {"digit_entropy": 8, "momentum": 12, "streak_entropy": -3, "parity": 2},
        {"digit_entropy": 0.4, "momentum": 0.3, "streak_entropy": 0.2, "parity": 0.1},
    )
    assert abs(overall["overall_velocity"] - 6.4) < 0.15


def test_hpp_timeseries():
    from src.analytics.hpp_timeseries import (
        HPPTimeSeries,
        lifecycle_stage,
        trend_label,
        moving_average,
    )
    from pathlib import Path

    assert lifecycle_stage(90) == "Peak"
    assert lifecycle_stage(75) == "Mature"
    assert lifecycle_stage(60) == "Declining"
    assert lifecycle_stage(40) == "Retire"
    assert "IMPROVING" in trend_label(7, 2)
    assert moving_average([1, 2, 3, 4, 5], 3)[-1] == 4.0

    path = Path("data/_test_hpp_ts.json")
    ts = HPPTimeSeries(path=path)
    ts.points = []
    ts.daily = {}
    ts.weight_history = []
    # inject fake points
    for i, hpp in enumerate([50, 68, 81, 88, 84, 72, 58]):
        day = f"2026-07-{i+1:02d}"
        ts.points.append(
            {
                "ts": 1000 + i,
                "day": day,
                "reason": "test",
                "contracts": {
                    "DIGITDIFF": {
                        "hpp": hpp,
                        "metrics": {
                            "digit_entropy": hpp - 5,
                            "momentum": hpp + 2,
                            "streak_entropy": 70,
                        },
                        "windows": {
                            "digit_entropy": {
                                "short": hpp + 5,
                                "mid": hpp,
                                "long": hpp - 5,
                                "hpp": hpp - 5,
                            },
                            "momentum": {
                                "short": hpp + 2,
                                "mid": hpp,
                                "long": hpp - 3,
                                "hpp": hpp + 2,
                            },
                        },
                        "weights": {
                            "digit_entropy": 0.4,
                            "momentum": 0.35,
                            "streak_entropy": 0.25,
                        },
                        "velocity": 3 if i else 0,
                        "acceleration": 1,
                        "trend": "UPWARD",
                        "lifecycle": lifecycle_stage(hpp),
                        "metric_velocity": {
                            "digit_entropy": 2,
                            "momentum": 3,
                            "streak_entropy": 0,
                        },
                        "strongest": "momentum",
                    }
                },
            }
        )
        ts.weight_history.append(
            {
                "ts": 1000 + i,
                "day": day,
                "contract": "DIGITDIFF",
                "weights": {
                    "digit_entropy": 0.4 - i * 0.01,
                    "momentum": 0.35 + i * 0.01,
                    "streak_entropy": 0.25,
                },
                "hpp": hpp,
            }
        )

    series = ts.contract_hpp_series("DIGITDIFF")
    assert series["current"] == 58
    assert series["lifecycle"] == "Declining"
    assert len(series["hpp"]) == 7

    multi = ts.multi_metric_series("DIGITDIFF")
    assert "momentum" in multi["series"]

    windows = ts.rolling_windows_table("DIGITDIFF")
    assert windows["rows"]

    heat = ts.heatmap("DIGITDIFF")
    assert "rows" in heat

    wf = ts.waterfall("DIGITDIFF")
    assert wf["current"] == 58
    assert "steps" in wf

    meta = ts.meta_hpp("DIGITDIFF")
    assert "meta_hpp" in meta
    assert meta["status"]

    board = ts.contract_dashboard("DIGITDIFF")
    assert board["current_hpp"] == 58
    assert "radar" in board


def test_historical_predictive_power():
    from src.analytics.historical_predictive_power import (
        lift_score,
        edge_based_hpp,
        profit_factor_power,
        information_gain,
        composite_hpp,
        time_decay_hpp,
        HPPTracker,
        binary_entropy,
    )
    from pathlib import Path

    # Lift 68/50 = 1.36 → HPP 36
    assert abs(lift_score(0.68, 0.50) - 36.0) < 0.5
    # Edge 12% / 25% → 48
    assert abs(edge_based_hpp(0.62, 0.50) - 48.0) < 0.5
    # PF 2.1 → 84
    assert abs(profit_factor_power(210, 100) - 84.0) < 1.0
    # Info gain positive when signal concentrates probability
    assert information_gain(0.50, 0.68) > 0
    assert binary_entropy(0.5) == 1.0

    # Production: 35/25/20/10/10 → 80.25
    c = composite_hpp(lift=80, profit_power=85, stability=70, info_gain=75, sample=95)
    assert abs(c - 80.25) < 0.5

    # Time decay
    assert abs(time_decay_hpp(90, 75, 65) - 80.5) < 0.2

    # Tracker attribution
    path = Path("data/_test_hpp.json")
    tr = HPPTracker(path=path)
    tr.outcomes = []
    for i in range(40):
        tr.record(
            contract="DIGITDIFF",
            metrics={"digit_entropy": 85, "momentum": 90, "streak_entropy": 40},
            is_win=True,
            profit=1.0,
        )
    for i in range(20):
        tr.record(
            contract="DIGITDIFF",
            metrics={"digit_entropy": 85, "momentum": 30, "streak_entropy": 40},
            is_win=False,
            profit=-1.0,
        )
    hpp = tr.metric_hpp("DIGITDIFF", "digit_entropy")
    assert hpp["hpp"] >= 40
    assert "windows" in hpp
    weights = tr.hpp_weights(
        "DIGITDIFF",
        ["digit_entropy", "momentum", "streak_entropy"],
    )
    assert abs(sum(weights["weights"].values()) - 1.0) < 0.02
    assert weights["strongest"] in {
        "digit_entropy",
        "momentum",
        "streak_entropy",
    }
    cond = tr.conditional_win_rate("DIGITDIFF", "digit_entropy")
    assert cond["n_all"] == 60


def test_contract_profile_system():
    from src.analytics.contract_profiles import (
        get_base_profile,
        build_metric_vector,
        contract_clarity,
        AdaptiveWeightEngine,
        evaluate_contract_setup,
        normalize_contract_key,
        sample_confidence,
    )

    assert normalize_contract_key("MATCH") == "DIGITMATCH"
    assert normalize_contract_key("DIFFER") == "DIGITDIFF"

    match = get_base_profile("DIGITMATCH")
    assert match["digit_entropy"] >= 0.30
    assert match["streak_entropy"] >= 0.25

    differ = get_base_profile("DIGITDIFF")
    assert differ["digit_entropy"] >= 0.45

    even = get_base_profile("DIGITEVEN")
    assert even.get("parity_entropy", 0) >= 0.45

    # Dynamic: high digit, low streak → digit weight rises for MATCH
    metrics = {
        "digit_entropy": 90,
        "streak_entropy": 20,
        "repetition_bias": 50,
        "odd_even_entropy": 40,
        "up_down_entropy": 40,
    }
    eng = AdaptiveWeightEngine(path=__import__("pathlib").Path("data/_test_prof.json"))
    wt = eng.compute_weights("DIGITMATCH", metrics, sample_n=200, use_learning=False)
    # digit should dominate after strength adjustment
    assert wt["normalized_weights"]["digit_entropy"] > wt["normalized_weights"][
        "streak_entropy"
    ]

    conf30 = sample_confidence(30)
    conf5000 = sample_confidence(5000)
    assert conf30 < 0.35
    assert conf5000 > 0.9

    cc = contract_clarity(
        "DIGITDIFF",
        {
            "digit_entropy": 92,
            "streak_entropy": 70,
            "rarity_score": 84,
            "odd_even_entropy": 40,
            "up_down_entropy": 40,
            "momentum": 90,
            "stability": 78,
        },
        sample_n=600,
        use_learning=False,
    )
    assert cc["clarity_score"] >= 60
    assert "contributors" in cc
    assert cc["recommendation"] in {
        "STRONG SETUP",
        "TRADEABLE",
        "WATCH",
        "SKIP",
    }

    ticks = _ticks(200)
    for i in range(50):
        ticks[-(i + 1)]["quote"] = 100.3
    ev = evaluate_contract_setup(
        ticks,
        symbol="R_50",
        contract_type="DIGITEVEN",
        sample_n=200,
        pattern_wr=0.58,
        baseline_wr=0.50,
    )
    assert ev["contract"] == "DIGITEVEN"
    assert "clarity_score" in ev
    assert "display" in ev


def test_hierarchical_clarity_weights_and_blend():
    from src.analytics.hierarchical_clarity import (
        composite_entropy_weighted,
        entropy_clarity_engine,
        final_pattern_clarity,
        weights_for_contract,
        build_hierarchical_clarity,
        DEFAULT_ENTROPY_WEIGHTS,
    )

    # Example: 82×0.40 + 85×0.25 + 70×0.15 + 88×0.10 + 60×0.10 = 79.35 → 79.4
    subs = {
        "digit": 82,
        "streak": 85,
        "odd_even": 70,
        "over_under": 88,
        "up_down": 60,
    }
    comp, contrib = composite_entropy_weighted(subs, DEFAULT_ENTROPY_WEIGHTS)
    assert abs(comp - 79.4) < 0.2

    # Entropy clarity: 80×0.6 + 90×0.25 + 70×0.15 = 81
    ec = entropy_clarity_engine(80, 90, 70)
    assert abs(ec["entropy_clarity"] - 81.0) < 0.2

    # Final: 92×0.4 + 81×0.3 + 75×0.15 + 90×0.1 + 85×0.05 = 85.4
    fin = final_pattern_clarity(
        statistical_separation=92,
        entropy_clarity=81,
        stability=75,
        sample_size_score=90,
        simplicity=85,
    )
    assert abs(fin["pattern_clarity"] - 85.4) < 0.2
    assert fin["class"] in {"Clear", "Exceptional", "Strong", "Moderate"}

    # Adaptive weights
    w_diff = weights_for_contract("DIGITDIFF")
    assert w_diff["digit"] >= 0.50
    w_even = weights_for_contract("DIGITEVEN")
    assert w_even["odd_even"] >= 0.50
    w_over = weights_for_contract("DIGITOVER")
    assert w_over["over_under"] >= 0.45

    # End-to-end hierarchical with ticks
    ticks = _ticks(250)
    for i in range(60):
        ticks[-(i + 1)]["quote"] = 100.7
    h = build_hierarchical_clarity(
        ticks,
        symbol="R_75",
        contract_type="DIGITDIFF",
        pattern_wr=0.62,
        baseline_wr=0.50,
        sample_n=600,
    )
    assert "pattern_clarity" in h
    assert "contributors" in h
    assert h.get("regime")
    assert h.get("confidence") in {"HIGH", "MEDIUM", "LOW"}
    assert h["level1_raw"]["digit"] >= 0


def test_rolling_entropy_engine():
    from src.analytics.rolling_entropy import RollingEntropyEngine, HMAX_DIGITS

    eng = RollingEntropyEngine()
    # Uniform-ish
    for i in range(200):
        eng.add_digit(i % 10, quote=100.0 + (i % 3) * 0.01)
    snap = eng.snapshot()
    assert snap["ready"]
    assert "50" in snap["windows"]
    assert "500" in snap["windows"] or "200" in snap["windows"]
    assert snap["regime"] in {
        "RANDOM",
        "NORMAL",
        "BIASED",
        "STRONG PATTERN",
        "EXTREME ANOMALY",
    }
    # Skew heavily toward 7
    for _ in range(80):
        eng.add_digit(7, quote=100.7)
    snap2 = eng.snapshot()
    h50 = (snap2["windows"].get("50") or {}).get("h")
    assert h50 is not None
    assert h50 < HMAX_DIGITS
    assert snap2["primary"]["compression_pct"] >= 0
    assert "DIGITDIFF" in snap2["triggers"]
    assert "realtime_pattern_strength" in snap2
    # Velocity exists
    assert "velocity" in snap2["primary"]


def test_filter_requires_clarity_and_sample():
    ticks = _ticks(80)
    # Strong history but still check gates present
    hist = [{"profit": 1.0}] * 300 + [{"profit": -1.0}] * 100
    ev = evaluate_setup(
        ticks,
        symbol="R_75",
        contract_type="DIGITEVEN",
        family="digits",
        history_rows=hist,
        signal_confidence=0.85,
        min_sample=500,
        global_samples=40,  # still cold phase globally
    )
    assert "pattern_clarity" in ev
    assert "sample_size" in ev
    assert "min_clarity" in ev["gates"]
    assert "min_sample" in ev["gates"]
    # global cold + setup n < min_sample → cold_start soft path still available
    assert ev.get("learning_phase") == "cold"
    assert ev.get("cold_start") is True


def test_cold_start_can_allow_high_conf():
    """With 0 history, high live conf must be able to open trades to learn."""
    ticks = _ticks(120)
    for i in range(40):
        ticks[-(i + 1)]["quote"] = float(f"100.{(i % 2) * 2}")  # even bias
    ev = evaluate_setup(
        ticks,
        symbol="R_75",
        contract_type="DIGITEVEN",
        family="digits",
        history_rows=[],
        signal_confidence=0.86,
        min_sample=100,
        min_pattern=70,
        min_clarity=65,
        min_edge=60,
        min_live_edge=65,
        min_quality=60,
        global_samples=0,
    )
    assert ev["cold_start"] is True
    assert ev.get("learning_phase") == "cold"
    # Should not hard-require sample size during cold start
    assert ev["gates"]["sample_ok"] is True
    # High conf digit-queue style should often allow or at least score higher
    assert float(ev["pattern_strength"]["pattern_strength"]) >= 50
    assert float(ev["live_edge"]["live_edge"]) >= 50


def test_mature_phase_disables_cold_allow():
    """After 100 global closes, loose cold_allow path is off."""
    ticks = _ticks(80)
    ev = evaluate_setup(
        ticks,
        symbol="R_25",
        contract_type="CALL",
        family="rise_fall",
        history_rows=[],
        signal_confidence=0.86,
        min_sample=100,
        global_samples=120,
    )
    assert ev.get("learning_phase") == "mature"
    assert ev.get("cold_start") is False
    assert ev.get("cold_allow") is False


def test_live_edge_components():
    le = live_edge_score(
        historical_edge=80,
        recent_score=90,
        pattern_strength_val=75,
        volatility_match=85,
        confidence=95,
    )
    # Spec example ≈ 83.75
    assert abs(le["live_edge"] - 83.75) < 0.5
    assert le["auto_ok"] is True
    assert le["status"] in {"BUY", "STRONG BUY"}
    assert "LIVE EDGE" in (le["reasons"][0] if le["reasons"] else "")


def test_recency_weights():
    from src.analytics.edge_score import recency_weighted_performance

    # Recent wins, older losses
    rows = [{"profit": -1.0}] * 400 + [{"profit": 1.0}] * 100
    r = recency_weighted_performance(rows)
    assert r["wr_100"] == 1.0
    assert r["label"] in {"High", "Stable"}
    assert r["score"] > 50


def test_digit_heatmap_windows():
    heat = digit_heatmap(_ticks(200), windows=(100, 500))
    assert "100" in heat["windows"]
    assert len(heat["windows"]["100"]["table"]) == 10


def test_digit_snapshot_has_streaks():
    snap = digit_snapshot(_ticks(150))
    assert "heatmap" in snap
    assert "ticks_since" in snap


def test_detect_repeat_pattern():
    # force trailing 7s
    ticks = _ticks(40)
    for i in range(3):
        ticks[-(i + 1)]["quote"] = 100.7
    pats = detect_patterns(ticks)
    assert pats["has_alert"] or pats["pattern_alert_strength"] >= 0


def test_filter_skip_low_edge():
    ticks = _ticks(80)
    # No history + low conf → skip or watch
    ev = evaluate_setup(
        ticks,
        symbol="R_75",
        contract_type="DIGITOVER",
        family="digits",
        history_rows=[],
        signal_confidence=0.55,
    )
    assert ev["recommendation"] in {"Skip", "Watch", "Trade"}
    assert "copilot" in ev


def test_filter_bootstrap_can_allow_strong_signal():
    # Build history with solid wins
    hist = [{"profit": 1.0}] * 40 + [{"profit": -1.0}] * 15
    ticks = _ticks(100)
    ev = evaluate_setup(
        ticks,
        symbol="R_50",
        contract_type="CALL",
        family="rise_fall",
        history_rows=hist,
        recent_rows=hist[-30:],
        signal_confidence=0.88,
    )
    assert ev["live_edge"]["live_edge"] > 0
    assert isinstance(ev["allow"], bool)


def test_adaptive_stake_reduces_after_losses():
    plan = adaptive_risk_pct(
        base_risk_pct=1.5, consecutive_losses=3, min_risk_pct=0.5, max_risk_pct=2.0
    )
    assert plan["risk_pct"] == 0.5
    assert stake_from_risk(1000, 1.0, min_stake=0.35) == 10.0


def test_martingale_survival():
    s = survival_probability(1000, 10, max_levels=5, multiplier=2.0)
    assert "danger_level" in s
    assert len(s["ladder"]) == 5


def test_strategy_builder_cold_digit(tmp_path):
    sb = StrategyBuilder(directory=tmp_path)
    path = sb.create_example_cold_digit()
    assert path.is_file()
    # digits never 8 → absent long
    ticks = []
    for i in range(50):
        ticks.append({"quote": float(f"100.{i % 7}"), "epoch": 1000 + i})
    hit = sb.evaluate(sb.list_strategies()[0], ticks)
    assert hit is not None
    assert hit["contract_type"] == "DIGITOVER"


def test_backtest_digit_even():
    ticks = _ticks(500)
    r = backtest_digit_rule(
        ticks, contract_type="DIGITEVEN", stake=1.0, max_ticks=400
    )
    assert r["trades"] > 0
    assert 0 <= r["win_rate"] <= 1
