"""
AI Trade Filter — never just "BUY NOW".

Outputs market condition, expected edge, recommendation (Trade / Skip / Watch).

Production auto-execution requires ALL of:
  Pattern Strength ≥ 75
  Pattern Clarity  ≥ 80
  Edge Score       ≥ 80   (historical EV-based score)
  Live Edge        ≥ 80
  Sample Size      ≥ 500
  Quality          ≥ 80   (composite setup quality)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.analytics.digit_analysis import digit_snapshot
from src.analytics.edge_score import (
    confidence_score_20,
    edge_label,
    historical_edge_score,
    live_edge_score,
    pattern_strength,
    recency_weighted_performance,
    stats_from_trade_rows,
)
from src.analytics.pattern_clarity import (
    baseline_for_contract,
    count_context_confirmations,
    pattern_clarity,
)
from src.analytics.tick_patterns import detect_patterns
from src.analytics.trade_quality import trade_quality_score
from src.analytics.no_trade_engine import evaluate_no_trade
from src.strategy.regime_filter import should_skip_digits, should_skip_rise_fall


# Auto-execution thresholds (override via env for Cloud Run testing vs production)
import os as _os


def _env_f(name: str, default: float) -> float:
    raw = _os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_i(name: str, default: int) -> int:
    raw = _os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Production defaults; Cloud Run can lower MIN_SAMPLE_SIZE so learning can start sooner
MIN_PATTERN_STRENGTH = _env_f("MIN_PATTERN_STRENGTH", 75.0)
MIN_PATTERN_CLARITY = _env_f("MIN_PATTERN_CLARITY", 80.0)
MIN_EDGE_SCORE = _env_f("MIN_EDGE_SCORE", 80.0)
MIN_LIVE_EDGE = _env_f("MIN_LIVE_EDGE", 80.0)
MIN_QUALITY = _env_f("MIN_QUALITY", 80.0)
# 500 = strict production; 80–100 = practical learning / demo on Cloud Run
MIN_SAMPLE_SIZE = _env_i("MIN_SAMPLE_SIZE", 500)
# After this many *global* closes, exit soft cold-start (require stricter gates)
COLD_START_EXIT_N = _env_i("COLD_START_EXIT_N", 50)
# After this many closes, use full production-ish thresholds
MATURE_SAMPLE_N = _env_i("MATURE_SAMPLE_N", 100)


def evaluate_setup(
    ticks: Sequence[Dict[str, Any]],
    *,
    symbol: str = "",
    contract_type: str = "",
    family: str = "digits",
    history_rows: Optional[Sequence[Dict[str, Any]]] = None,
    recent_rows: Optional[Sequence[Dict[str, Any]]] = None,
    signal_confidence: float = 0.0,
    barrier: Optional[int] = None,
    n_rule_conditions: int = 2,
    pattern_frequency: Optional[float] = None,
    min_pattern: float = MIN_PATTERN_STRENGTH,
    min_clarity: float = MIN_PATTERN_CLARITY,
    min_edge: float = MIN_EDGE_SCORE,
    min_live_edge: float = MIN_LIVE_EDGE,
    min_quality: float = MIN_QUALITY,
    min_sample: int = MIN_SAMPLE_SIZE,
    bootstrap_n: int = 30,
    global_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full filter evaluation for one candidate setup.

    global_samples: total closed trades across all setups (learner). Used to
    tighten cold-start after COLD_START_EXIT_N / MATURE_SAMPLE_N.
    """
    history_rows = list(history_rows or [])
    recent_rows = list(recent_rows or history_rows[-100:])
    hist_stats = stats_from_trade_rows(history_rows)
    recent_stats = stats_from_trade_rows(recent_rows)
    n_samples = int(hist_stats["n"])
    g_n = int(global_samples) if global_samples is not None else n_samples

    hist = historical_edge_score(
        wins=hist_stats["wins"],
        losses=hist_stats["losses"],
        gross_profit=hist_stats["gross_profit"],
        gross_loss=hist_stats["gross_loss"],
        max_dd_pct=hist_stats["max_dd_pct"],
    )

    # Cold-start: soft scores for display, but sample gate still blocks auto-exec
    bootstrapped = False
    if n_samples < bootstrap_n and signal_confidence > 0:
        bootstrapped = True
        live_prior = max(0.0, min(100.0, float(signal_confidence) * 100.0))
        blended = 0.45 * float(hist["edge_score"]) + 0.55 * live_prior * 0.85
        hist = {
            **hist,
            "edge_score": round(blended, 1),
            "label": edge_label(blended),
            "reasons": list(hist.get("reasons") or [])
            + [f"Bootstrap blend (n={n_samples}<{bootstrap_n})"],
        }

    recency = recency_weighted_performance(history_rows)
    pats = detect_patterns(ticks)

    # Pattern WR (bayesian when possible)
    raw_wr = (
        hist_stats["wins"] / n_samples if n_samples else float(signal_confidence or 0.5)
    )
    bayes_wr = float(hist["metrics"].get("bayes_wr") or raw_wr)
    baseline = baseline_for_contract(contract_type, barrier)

    # Context confirmations (same-direction support)
    n_conf, conf_notes = count_context_confirmations(
        ticks, contract_type=contract_type, family=family
    )

    # Frequency: rare alerts → low frequency; estimate from pattern strength alerts
    if pattern_frequency is not None:
        freq = max(0.0, min(1.0, float(pattern_frequency)))
    elif pats.get("has_alert"):
        # Stronger/rarer alerts → lower frequency
        strength = float(pats.get("pattern_alert_strength") or 50)
        freq = max(0.02, min(0.35, 0.25 - strength / 500.0))
    else:
        freq = 0.18  # somewhat common default

    # Contract Profile System + hierarchical / rolling entropy
    rolling: Dict[str, Any] = {}
    hier: Dict[str, Any] = {}
    profile_eval: Dict[str, Any] = {}
    try:
        from src.analytics.contract_profiles import evaluate_contract_setup

        profile_eval = evaluate_contract_setup(
            list(ticks)[-500:],
            symbol=symbol or "_default",
            contract_type=str(contract_type or "DIGITDIFF"),
            sample_n=n_samples,
            pattern_wr=bayes_wr if n_samples else max(0.5, float(signal_confidence or 0.5)),
            baseline_wr=baseline,
        )
        rolling = profile_eval.get("rolling") or {}
    except Exception:
        profile_eval = {}

    try:
        from src.analytics.hierarchical_clarity import build_hierarchical_clarity

        hier = build_hierarchical_clarity(
            list(ticks)[-500:],
            symbol=symbol or "_default",
            contract_type=str(contract_type or ""),
            pattern_wr=bayes_wr if n_samples else max(0.5, float(signal_confidence or 0.5)),
            baseline_wr=baseline,
            sample_n=n_samples,
            n_conditions=n_rule_conditions,
        )
        if not rolling:
            rolling = hier.get("rolling") or {}
    except Exception:
        if not rolling:
            try:
                from src.analytics.rolling_entropy import feed_ticks

                rolling = feed_ticks(symbol or "_default", list(ticks)[-500:])
            except Exception:
                rolling = {}

    # Prefer contract-profile clarity when available; blend with hierarchical
    if profile_eval.get("clarity_score") is not None:
        prof_c = float(profile_eval["clarity_score"])
        hier_c = float(hier.get("pattern_clarity") or prof_c)
        # Blend: profile drives contract-specific score; hierarchical adds structure
        blended = 0.55 * prof_c + 0.45 * hier_c
        from src.analytics.pattern_clarity import clarity_class

        clarity = {
            "pattern_clarity": round(blended, 1),
            "class": clarity_class(blended),
            "auto_ok": blended >= 80
            and float(profile_eval.get("sample_confidence") or 0) >= 0.45,
            "formula": "contract_profile+hierarchical",
            "profile_clarity": prof_c,
            "hierarchical_clarity": hier_c,
            "recommendation": profile_eval.get("recommendation"),
            "reasons": list(profile_eval.get("reasons") or [])
            + list((hier.get("reasons") or [])[:4]),
            "explain": profile_eval.get("explain") or profile_eval.get("reasons"),
            "contributors": profile_eval.get("contributors")
            or hier.get("contributors")
            or [],
            "weights": profile_eval.get("weights"),
            "weight_detail": profile_eval.get("weight_detail"),
            "base_profile": profile_eval.get("base_profile"),
            "metrics": profile_eval.get("metrics"),
            "regime": profile_eval.get("regime") or rolling.get("regime"),
            "confidence": profile_eval.get("confidence") or hier.get("confidence"),
            "confidence_score": profile_eval.get("sample_confidence"),
            "display": profile_eval.get("display"),
            "rolling_entropy": rolling,
            "contract_profile": profile_eval,
            "level1_raw": hier.get("level1_raw"),
            "level2_composite": hier.get("level2_composite"),
            "level3_final": hier.get("level3_final"),
            "components": {
                **(hier.get("components") or {}),
                "profile_clarity": prof_c,
                "blended_clarity": blended,
            },
        }
    elif hier.get("pattern_clarity") is not None:
        clarity = {
            "pattern_clarity": hier["pattern_clarity"],
            "class": hier.get("class"),
            "auto_ok": hier.get("auto_ok"),
            "formula": "hierarchical",
            "reasons": hier.get("reasons") or [],
            "explain": hier.get("explain") or hier.get("reasons") or [],
            "components": hier.get("components") or {},
            "contributors": hier.get("contributors") or [],
            "regime": hier.get("regime"),
            "confidence": hier.get("confidence"),
            "confidence_score": hier.get("confidence_score"),
            "display": hier.get("display"),
            "level1_raw": hier.get("level1_raw"),
            "level2_composite": hier.get("level2_composite"),
            "level3_final": hier.get("level3_final"),
            "rolling_entropy": rolling,
            "entropy_strength_detail": hier.get("entropy_strength_detail"),
        }
    else:
        clarity = pattern_clarity(
            pattern_wr=bayes_wr if n_samples else max(0.5, float(signal_confidence or 0.5)),
            baseline_wr=baseline,
            frequency=freq,
            trade_rows=history_rows,
            confirmations=n_conf,
            n_conditions=n_rule_conditions,
            explainable=True,
            ticks=ticks,
            sample_n=n_samples,
            formula="production",
        )
        if rolling.get("regime"):
            clarity = {
                **clarity,
                "regime": rolling.get("regime"),
                "rolling_entropy": rolling,
            }

    ps = pattern_strength(
        wins=hist_stats["wins"],
        losses=hist_stats["losses"],
        recent_wins=recent_stats["wins"],
        recent_losses=recent_stats["losses"],
        rarity_score=float(clarity["pattern_clarity"]),
        clarity_score=float(clarity["pattern_clarity"]),
        profit_factor=(
            float(hist["metrics"]["profit_factor"]) if hist.get("metrics") else None
        ),
        formula="production",
    )

    # Cold-start phases (global closes matter more than single-setup n):
    #   cold    g_n < 50  → soft live proxies so history can build
    #   warming 50–99     → half-strict (pattern/clarity raised)
    #   mature  ≥100      → no live soft path; require real gates
    phase = "cold"
    if g_n >= MATURE_SAMPLE_N:
        phase = "mature"
    elif g_n >= COLD_START_EXIT_N:
        phase = "warming"

    # Per-setup cold only while globally still learning AND setup under sample
    cold_start = phase != "mature" and n_samples < max(int(min_sample), 30)
    if phase == "mature":
        cold_start = False
    # After exit N, never use the most aggressive cold_allow path
    rt_ps = float(rolling.get("realtime_pattern_strength") or 0)
    live_sig = max(0.0, min(1.0, float(signal_confidence or 0)))
    if cold_start and live_sig > 0 and phase == "cold":
        # Historical edge score with n=0 is near-useless — seed from live conf
        live_edge_seed = live_sig * 100.0 * 0.80
        hist = {
            **hist,
            "edge_score": round(
                max(float(hist.get("edge_score") or 0), live_edge_seed), 1
            ),
            "bootstrapped": True,
        }
        # Live pattern proxy 0–100
        live_pattern = max(
            rt_ps,
            live_sig * 100.0 * 0.85,
            float(clarity.get("pattern_clarity") or 0) * 0.5 + live_sig * 40.0,
        )
        ps = {
            **ps,
            "pattern_strength": round(
                max(float(ps["pattern_strength"]), live_pattern * 0.9), 1
            ),
            "class": ps.get("class"),
            "bootstrapped": True,
            "reasons": list(ps.get("reasons") or [])
            + [f"Cold-start live pattern proxy {live_pattern:.0f} (n={n_samples})"],
        }
        # Boost clarity so hierarchical near-zero doesn't lock out forever
        if float(clarity.get("pattern_clarity") or 0) < 70:
            boosted = max(
                float(clarity.get("pattern_clarity") or 0),
                live_sig * 82.0,
                52.0 if live_sig >= 0.72 else 0.0,
            )
            clarity = {
                **clarity,
                "pattern_clarity": round(boosted, 1),
                "bootstrapped": True,
            }
    elif cold_start and phase == "warming" and live_sig > 0:
        # Milder boost only — require real structure to form
        live_pattern = max(rt_ps, live_sig * 70.0)
        ps = {
            **ps,
            "pattern_strength": round(
                max(float(ps["pattern_strength"]), live_pattern * 0.55), 1
            ),
            "warming": True,
        }

    # Tighten required thresholds after cold-start exit
    eff_min_pattern = min_pattern
    eff_min_clarity = min_clarity
    eff_min_edge = min_edge
    eff_min_live = min_live_edge
    eff_min_quality = min_quality
    if phase == "warming":
        # Halfway from soft env defaults toward production
        eff_min_pattern = max(min_pattern, 72.0)
        eff_min_clarity = max(min_clarity, 72.0)
        eff_min_edge = max(min_edge, 70.0)
        eff_min_live = max(min_live_edge, 72.0)
        eff_min_quality = max(min_quality, 70.0)
    elif phase == "mature":
        eff_min_pattern = max(min_pattern, 75.0)
        eff_min_clarity = max(min_clarity, 75.0)
        eff_min_edge = max(min_edge, 75.0)
        eff_min_live = max(min_live_edge, 78.0)
        eff_min_quality = max(min_quality, 75.0)

    skip_d, _, dig_reg = should_skip_digits(ticks)
    skip_rf, _, rf_reg = should_skip_rise_fall(ticks)
    chop = float((dig_reg or rf_reg or {}).get("chop_score") or 0.0)
    vol_match = max(15.0, 95.0 - chop * 80.0)
    if family == "digits" and skip_d:
        vol_match = min(vol_match, 35.0)
    if family in {"rise_fall", "minute_rise_fall"} and skip_rf:
        vol_match = min(vol_match, 35.0)

    conf_pts = confidence_score_20(n_samples) / 20.0 * 100.0
    if bootstrapped or cold_start:
        conf_pts = max(conf_pts, live_sig * 80.0)

    live = live_edge_score(
        historical_edge=float(hist["edge_score"]),
        recent_score=float(recency["score"]),
        pattern_strength_val=float(ps["pattern_strength"]),
        volatility_match=vol_match,
        confidence=conf_pts,
        extra_reasons=list(hist.get("reasons") or [])[:4]
        + list(ps.get("reasons") or [])[:2]
        + list(clarity.get("reasons") or [])[:2],
    )
    if cold_start and phase == "cold" and live_sig >= 0.78:
        # Lift live edge only in true cold phase
        lifted = max(
            float(live["live_edge"]),
            live_sig * 85.0 * 0.7 + rt_ps * 0.25,
        )
        live = {**live, "live_edge": round(min(95.0, lifted), 1), "bootstrapped": True}

    quality = trade_quality_score(
        ticks,
        symbol=symbol,
        contract_type=contract_type,
        historical_edge=float(hist["edge_score"]),
        pattern_strength_val=float(ps["pattern_strength"]),
        live_edge=float(live["live_edge"]),
    )
    if cold_start and phase == "cold" and live_sig >= 0.78:
        quality = {
            **quality,
            "quality_score": round(
                max(float(quality["quality_score"]), live_sig * 80.0), 1
            ),
        }

    # --- Gates (tighten after global sample milestones) ---
    p_ok = float(ps["pattern_strength"]) >= eff_min_pattern
    c_ok = float(clarity["pattern_clarity"]) >= eff_min_clarity
    edge_ok = float(hist["edge_score"]) >= eff_min_edge
    e_ok = float(live["live_edge"]) >= eff_min_live
    q_ok = float(quality["quality_score"]) >= eff_min_quality
    # Sample gate: only soft while cold phase
    s_ok = n_samples >= min_sample if phase == "mature" else True

    # Entropy-based contract family triggers — soft during cold only
    ct_up = str(contract_type or "").upper()
    triggers = (rolling.get("triggers") or {}) if rolling else {}
    trig = triggers.get(ct_up) or {}
    entropy_trig_ok = True
    if family == "digits" and ct_up in triggers and phase != "cold":
        entropy_trig_ok = bool(trig.get("allow"))
        comp_pct = float((rolling.get("primary") or {}).get("compression_pct") or 0)
        if ct_up == "DIGITDIFF":
            entropy_trig_ok = comp_pct > 15 and (
                rt_ps > 75 or float(ps["pattern_strength"]) > 75
            )
        elif ct_up == "DIGITMATCH":
            entropy_trig_ok = comp_pct > 20 and bool(trig.get("allow"))

    allow = (
        p_ok
        and c_ok
        and edge_ok
        and e_ok
        and q_ok
        and s_ok
        and entropy_trig_ok
    )

    # Explicit cold-start allow ONLY in cold phase (not forever).
    # Thresholds intentionally lower than mature so n=0 history can start.
    cold_allow = False
    if cold_start and phase == "cold" and not allow:
        cold_allow = (
            live_sig >= 0.72
            and chop < 0.78
            and float(quality["quality_score"]) >= 50
            and float(live["live_edge"]) >= 50
            and float(ps["pattern_strength"]) >= 50
            and float(clarity.get("pattern_clarity") or 0) >= 45
        )
        if cold_allow:
            allow = True

    # Market category → which scoring engine weights matter
    market_profile: Dict[str, Any] = {}
    try:
        from src.strategy.market_categories import market_profile as _mprof

        market_profile = _mprof(symbol or "")
    except Exception:
        market_profile = {}

    # --- Second system: Momentum + Persistence + Transition ---
    mp_analysis: Dict[str, Any] = {}
    try:
        from src.analytics.momentum_persistence_engine import (
            analyze_momentum_persistence,
            dual_system_blend,
        )

        mp_analysis = analyze_momentum_persistence(
            list(ticks)[-200:],
            symbol=symbol or "",
            contract_type=ct_up or str(contract_type or ""),
            note_velocity=True,
        )
        # Digit trades: MP as secondary confirmation → confidence delta
        dig_conf = mp_analysis.get("digit_confirmation") or {}
        if family == "digits" and dig_conf.get("confidence_delta"):
            live_sig = max(
                0.0,
                min(1.0, live_sig + float(dig_conf["confidence_delta"]) / 100.0),
            )
    except Exception:
        mp_analysis = {}

    # --- Rise/Fall directional reweight (momentum / persistence / vol / dir-entropy) ---
    rf_analysis: Dict[str, Any] = {}
    is_rf = family in {"rise_fall", "minute_rise_fall"} or ct_up in {
        "CALL",
        "PUT",
        "RISE",
        "FALL",
    }
    if is_rf:
        try:
            from src.analytics.rise_fall_engine import (
                analyze_rise_fall,
                rf_pattern_clarity,
                rf_pattern_strength,
            )

            rf_analysis = profile_eval.get("rise_fall") or analyze_rise_fall(
                list(ticks)[-200:],
                contract_type=ct_up or "CALL",
                hpp=float(hist.get("edge_score") or 50),
            )
            rf_ps = rf_pattern_strength(rf_analysis)
            rf_cl = rf_pattern_clarity(rf_analysis)
            # Reweight pattern strength / clarity toward directional model
            ps = {
                **ps,
                "pattern_strength": round(
                    0.35 * float(ps.get("pattern_strength") or 0)
                    + 0.65 * rf_ps,
                    1,
                ),
                "rf_score": rf_analysis.get("rf_score"),
                "family_model": "rise_fall",
            }
            clarity = {
                **clarity,
                "pattern_clarity": round(
                    0.30 * float(clarity.get("pattern_clarity") or 0)
                    + 0.70 * rf_cl,
                    1,
                ),
                "rf_clarity": rf_cl,
                "family_model": "rise_fall",
            }
            # Block RF when volatility is expanding/chaotic (unless cold soft)
            if not rf_analysis.get("vol_tradeable", True) and not cold_start:
                allow = False
        except Exception:
            rf_analysis = {}

    # --- No-Trade / EV decision engine (elite: when NOT to trade) ---
    no_trade = _run_no_trade_engine(
        contract_type=ct_up or str(contract_type or ""),
        family=family,
        pattern_clarity=float(clarity.get("pattern_clarity") or 0),
        pattern_strength=float(ps.get("pattern_strength") or 0),
        quality_setup=float(quality.get("quality_score") or 0),
        live_edge=float(live.get("live_edge") or 0),
        signal_confidence=live_sig,
        rolling=rolling,
        profile_eval=profile_eval,
        hist_stats=hist_stats,
        hist=hist,
        cold_start=cold_start,
        barrier=barrier,
        n_samples=n_samples,
        rf_analysis=rf_analysis,
        mp_analysis=mp_analysis,
    )
    # Meta-validator: strength, clarity, HPP, velocity, EV, conf, regime must agree
    meta = no_trade.get("meta_validator") or {}
    # Final allow requires classic gates AND no-trade ALLOW AND meta APPROVED
    if allow and not no_trade.get("allow"):
        allow = False
        cold_allow = False
    if allow and meta and not meta.get("allow"):
        allow = False
        cold_allow = False
    # If classic gates blocked but no-trade cold-start soft path opened, keep blocked
    # unless classic already allowed via cold_allow
    if not allow and no_trade.get("allow") and cold_start and cold_allow:
        if not meta or meta.get("allow"):
            allow = True

    if chop > 0.55:
        condition = "Choppy"
    elif chop > 0.35:
        condition = "Neutral"
    else:
        condition = "Trending" if family != "digits" else "Stable digits"

    edge_word = edge_label(float(live["live_edge"]))
    if float(live["live_edge"]) < 60:
        expected_edge = "Low"
    elif float(live["live_edge"]) < 80:
        expected_edge = "Moderate"
    else:
        expected_edge = "High"

    if allow:
        recommendation = "Trade"
        action = "ALLOW"
    elif (
        float(live["live_edge"]) >= 70
        and float(ps["pattern_strength"]) >= 65
        and float(clarity["pattern_clarity"]) >= 65
        and no_trade.get("status") != "REJECTED"
    ):
        recommendation = "Watch"
        action = "WATCH"
    else:
        recommendation = "Skip"
        action = "SKIP"
    if not allow and no_trade.get("status") == "REJECTED":
        recommendation = "Skip"
        action = "REJECTED"

    reasons: List[str] = []
    reasons.append(
        f"Edge Score: {hist.get('edge_score')} ({hist.get('label')}) · "
        f"Live Edge: {live.get('live_edge')} ({live.get('status')}) · "
        f"Clarity: {clarity.get('pattern_clarity')} ({clarity.get('class')})"
    )
    if cold_start:
        reasons.append(
            f"Cold-start mode (n={n_samples} < {max(int(min_sample), 30)}) — "
            "using live signal + entropy proxies so learning can collect samples"
        )
    if cold_allow:
        reasons.append("✓ Cold-start LIVE allow (high conf setup while building history)")
    # No-trade / EV summary
    nt_disp = no_trade.get("display") or {}
    reasons.append(
        f"No-Trade Engine: {no_trade.get('status')} · "
        f"TQ={((no_trade.get('trade_quality') or {}).get('trade_quality'))} · "
        f"EV={no_trade.get('ev')} · Regime={no_trade.get('regime')} · "
        f"Risk%={no_trade.get('risk_pct')}"
    )
    if no_trade.get("blocks"):
        reasons.append(f"✗ BLOCKED: {no_trade.get('reason')}")
    elif nt_disp.get("status") == "ALLOWED":
        reasons.append("✓ No-trade gates passed (clarity/velocity/stability/EV/ensemble)")
    if meta:
        reasons.append(
            f"Meta-Validator: {meta.get('status')} · "
            f"{meta.get('n_ok')}/{meta.get('n_total')} checks · "
            f"{meta.get('reason')}"
        )
    if mp_analysis:
        mpd = mp_analysis.get("display") or {}
        pvel = mp_analysis.get("persistence_velocity") or {}
        reasons.append(
            f"Momentum+Persistence: MP={mp_analysis.get('mp_score')} · "
            f"mom={mpd.get('momentum')} · persist={mpd.get('persistence')} · "
            f"engine={mp_analysis.get('persistence_engine_score')} · "
            f"conf={mpd.get('sample_conf')}"
        )
        reasons.append(
            f"Persistence Velocity: {pvel.get('interpretation') or mpd.get('interpretation') or '—'} · "
            f"MTF {mpd.get('fast_med_slow') or '—'} · "
            f"vel={pvel.get('velocity')} accel={pvel.get('acceleration')} "
            f"score={pvel.get('velocity_score')}"
        )
        dig = mp_analysis.get("digit_confirmation") or {}
        if dig.get("note"):
            reasons.append(f"Digit MP confirm: {dig['note']}")
    if rf_analysis:
        disp = rf_analysis.get("display") or {}
        reasons.append(
            f"Rise/Fall model: score={rf_analysis.get('rf_score')} · "
            f"mom={disp.get('momentum')} · vol={disp.get('vol_regime')} · "
            f"persist={disp.get('persistence')} · dirH={disp.get('dir_entropy')}"
        )

    def _gate(ok: bool, name: str, val: float, need: float) -> None:
        mark = "✓" if ok else "✗"
        reasons.append(f"{mark} {name} {val:.0f} {'≥' if ok else '<'} {need:.0f}")

    reasons.append(
        f"Learning phase={phase} global_samples={g_n} "
        f"(cold_exit={COLD_START_EXIT_N} mature={MATURE_SAMPLE_N})"
    )
    _gate(p_ok, "Pattern strength", float(ps["pattern_strength"]), eff_min_pattern)
    _gate(c_ok, "Pattern clarity", float(clarity["pattern_clarity"]), eff_min_clarity)
    _gate(edge_ok, "Edge score", float(hist["edge_score"]), eff_min_edge)
    _gate(e_ok, "Live edge", float(live["live_edge"]), eff_min_live)
    _gate(q_ok, "Quality", float(quality["quality_score"]), eff_min_quality)
    if phase == "cold":
        reasons.append(
            f"~ Sample size {n_samples} (cold-start; building toward {min_sample})"
        )
    elif phase == "warming":
        reasons.append(
            f"~ Warming phase n_global={g_n} — stricter gates, no loose cold_allow"
        )
    elif s_ok:
        reasons.append(f"✓ Sample size {n_samples} ≥ {min_sample}")
    else:
        reasons.append(
            f"✗ Sample size {n_samples} < {min_sample} — not enough for auto-exec"
        )
    if family == "digits" and ct_up:
        mark = "✓" if entropy_trig_ok else "✗"
        reasons.append(
            f"{mark} Entropy trigger {ct_up}: "
            f"{(trig.get('reason') if trig else 'n/a')}"
        )
    if rolling.get("regime"):
        reasons.append(
            f"Regime {rolling.get('regime')} · "
            f"RT pattern strength {rolling.get('realtime_pattern_strength')} · "
            f"composite entropy {rolling.get('composite_entropy')}"
        )

    for line in (clarity.get("reasons") or [])[:4]:
        if line not in reasons:
            reasons.append(line)
    for line in (hist.get("reasons") or [])[:3]:
        if line not in reasons:
            reasons.append(line)
    if conf_notes:
        reasons.append("Context: " + ", ".join(conf_notes[:4]))
    if recency.get("label"):
        reasons.append(
            f"Recency: {recency.get('label')} (weighted WR={recency.get('weighted_wr')})"
        )

    snap = digit_snapshot(ticks)
    copilot = _copilot_blurb(
        symbol=symbol,
        contract_type=contract_type,
        snap=snap,
        ps=ps,
        live=live,
        hist=hist,
        clarity=clarity,
        recommendation=recommendation,
        n_samples=n_samples,
    )

    decision_tq = (no_trade.get("trade_quality") or {}).get("trade_quality")
    return {
        "allow": allow,
        "action": action,
        "recommendation": recommendation,
        "market_condition": condition,
        "expected_edge": expected_edge,
        "edge_label": edge_word,
        "pattern_strength": ps,
        "pattern_clarity": clarity,
        "historical_edge": hist,
        "live_edge": live,
        "recency": recency,
        "quality": quality,
        "patterns": pats,
        "context_confirmations": n_conf,
        "context_notes": conf_notes,
        "baseline_wr": baseline,
        "sample_size": n_samples,
        "reasons": reasons,
        "explain": reasons,
        "copilot": copilot,
        "bootstrapped": bootstrapped,
        "rolling_entropy": rolling,
        "entropy_regime": rolling.get("regime"),
        "entropy_display": rolling.get("display"),
        "cold_start": cold_start,
        "cold_allow": cold_allow,
        "learning_phase": phase,
        "global_samples": g_n,
        # Decision engine outputs
        "no_trade": no_trade,
        "meta_validator": meta,
        "momentum_persistence": mp_analysis or None,
        "mp_score": (mp_analysis or {}).get("mp_score"),
        "rise_fall": rf_analysis or None,
        "decision_quality": decision_tq,
        "ev": no_trade.get("ev"),
        "p_win": no_trade.get("p_win"),
        "risk_pct": no_trade.get("risk_pct"),
        "regime": no_trade.get("regime"),
        "ensemble": no_trade.get("ensemble"),
        "edge_decay_pct": no_trade.get("edge_decay_pct"),
        "hpp": no_trade.get("_hpp"),
        "hpp_velocity": no_trade.get("_hpp_velocity"),
        "gates": {
            "min_pattern": min_pattern,
            "min_clarity": min_clarity,
            "min_edge": min_edge,
            "min_live_edge": min_live_edge,
            "min_quality": min_quality,
            "min_sample": min_sample,
            "pattern_ok": p_ok,
            "clarity_ok": c_ok,
            "edge_ok": edge_ok,
            "live_ok": e_ok,
            "quality_ok": q_ok,
            "sample_ok": s_ok,
            "entropy_trigger_ok": entropy_trig_ok,
            "no_trade_ok": bool(no_trade.get("allow")),
            "meta_ok": bool(meta.get("allow")) if meta else True,
            "ev_ok": float(no_trade.get("ev") or 0) > 0,
            "rf_ok": (
                bool(rf_analysis.get("vol_tradeable", True))
                if rf_analysis
                else True
            ),
            "cold_start": cold_start,
            "cold_allow": cold_allow,
            "learning_phase": phase,
            "eff_min_pattern": eff_min_pattern,
            "eff_min_clarity": eff_min_clarity,
        },
        "symbol": symbol,
        "contract_type": contract_type,
        "family": family,
        "market_category": market_profile.get("category"),
        "scoring_path": market_profile.get("scoring_path"),
        "market_profile": {
            "category": market_profile.get("category"),
            "label": market_profile.get("label"),
            "primary_metrics": market_profile.get("primary_metrics"),
            "engine_weights": market_profile.get("engine_weights"),
            "allowed_contracts": market_profile.get("allowed_contracts"),
        }
        if market_profile
        else None,
    }


def _run_no_trade_engine(
    *,
    contract_type: str,
    family: str,
    pattern_clarity: float,
    pattern_strength: float,
    quality_setup: float,
    live_edge: float,
    signal_confidence: float,
    rolling: Dict[str, Any],
    profile_eval: Dict[str, Any],
    hist_stats: Dict[str, Any],
    hist: Dict[str, Any],
    cold_start: bool,
    barrier: Optional[int],
    n_samples: int,
    rf_analysis: Optional[Dict[str, Any]] = None,
    mp_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build inputs for evaluate_no_trade from rolling entropy + HPP + history.
    Injects Momentum Persistence into Final Trade Quality.
    """
    rf_analysis = rf_analysis or {}
    mp_analysis = mp_analysis or {}
    rt_ps = float(rolling.get("realtime_pattern_strength") or 0)
    if rf_analysis.get("rf_score") is not None:
        rt_ps = max(rt_ps, float(rf_analysis["rf_score"]))
    stability = float(rolling.get("stability_score") or 55.0)
    if rf_analysis.get("volatility"):
        stability = max(
            stability * 0.4,
            float((rf_analysis["volatility"] or {}).get("score") or stability),
        )
    regime_raw = (
        rolling.get("regime")
        or profile_eval.get("regime")
        or "RANDOM"
    )
    # Map vol chaotic/expanding into a hostile decision regime for RF
    vol_reg = (rf_analysis.get("volatility") or {}).get("regime")
    if vol_reg in {"CHAOTIC", "EXPANDING"} and family in {
        "rise_fall",
        "minute_rise_fall",
    }:
        # Keep underlying market regime but no_trade will see vol via meta
        pass

    # HPP: prefer learned strongest HPP; fallback to composite of live metrics
    hpp = 55.0
    peak_hpp: Optional[float] = None
    hpp_velocity = 0.0
    try:
        from src.analytics.contract_profiles import get_weight_engine

        eng = get_weight_engine()
        metrics = profile_eval.get("metrics") or {}
        if metrics:
            rep = eng.hpp_report(contract_type, list(metrics.keys()))
            if rep.get("strongest_hpp") is not None:
                hpp = float(rep["strongest_hpp"])
            by_m = rep.get("hpp_by_metric") or {}
            if by_m:
                hpp = max(hpp, max(float(v) for v in by_m.values()))
        # Live proxy when no outcomes: average metric vector * slight edge blend
        if n_samples < 30 and metrics:
            live_hpp = sum(float(v) for v in metrics.values()) / max(1, len(metrics))
            hpp = max(hpp, live_hpp * 0.85 + float(live_edge) * 0.15)
    except Exception:
        pass

    # Blend live edge / pattern into HPP when cold
    if cold_start:
        hpp = max(
            hpp,
            0.45 * float(pattern_strength)
            + 0.30 * float(pattern_clarity)
            + 0.25 * float(live_edge),
        )

    try:
        from src.analytics.calibration import get_calibration_tracker

        cal = get_calibration_tracker()
        peak_hpp = cal.peak_hpp(contract_type) or None
        if peak_hpp is not None and peak_hpp > 0:
            cal.note_hpp(contract_type, hpp)
            peak_hpp = max(float(peak_hpp), float(hpp))
        else:
            peak_hpp = cal.note_hpp(contract_type, hpp)
    except Exception:
        peak_hpp = None

    # Velocity from HPP time series if available; else rolling entropy velocity
    try:
        from src.analytics.hpp_timeseries import get_hpp_timeseries

        ts = get_hpp_timeseries()
        snap = getattr(ts, "latest", None) or {}
        if callable(getattr(ts, "snapshot", None)):
            try:
                snap = ts.snapshot() or snap
            except Exception:
                pass
        contracts = (snap or {}).get("contracts") or {}
        cdat = contracts.get(str(contract_type).upper()) or {}
        vel = cdat.get("overall_velocity")
        if vel is None:
            vel = (cdat.get("velocity") or {}).get("overall_velocity")
        if vel is not None:
            hpp_velocity = float(vel)
        elif rolling.get("primary"):
            # map entropy velocity (bits) into roughly ±15 scale
            raw_v = float((rolling.get("primary") or {}).get("velocity") or 0)
            hpp_velocity = max(-15.0, min(15.0, -raw_v * 50.0))
    except Exception:
        raw_v = float((rolling.get("primary") or {}).get("velocity") or 0)
        hpp_velocity = max(-15.0, min(15.0, -raw_v * 50.0))

    # Reward: net win multiple; use bayes WR to refine p_win
    reward = 0.92
    ct = str(contract_type or "").upper()
    if ct in {"DIGITDIFF"}:
        reward = 0.09  # ~1.09 payout common on differ? actually differ often ~0.95–1.0
        # DIGITDIFF wins often (~90%) with low payout; EV uses p_win carefully
        reward = 0.095
    elif ct in {"DIGITEVEN", "DIGITODD"}:
        reward = 0.95
    elif ct in {"CALL", "PUT"}:
        reward = 0.95

    p_win = None
    bayes = (hist.get("metrics") or {}).get("bayes_wr")
    if bayes is not None and n_samples >= 20:
        p_win = float(bayes)
    elif signal_confidence > 0:
        p_win = 0.50 * float(signal_confidence) + 0.50 * (
            0.55 * pattern_strength / 100.0 + 0.45 * pattern_clarity / 100.0
        )

    conf_100 = max(
        float(signal_confidence) * 100.0,
        float(rolling.get("confidence") or 0),
        quality_setup,
    )
    # Digit MP confirmation already adjusted signal_confidence; apply to conf_100
    dig_delta = float(
        ((mp_analysis.get("digit_confirmation") or {}).get("confidence_delta") or 0)
    )
    conf_100 = max(0.0, min(100.0, conf_100 + dig_delta))

    mp_score = None
    if mp_analysis.get("mp_score") is not None:
        mp_score = float(mp_analysis["mp_score"])
    elif (mp_analysis.get("momentum_persistence") or {}).get("momentum_persistence") is not None:
        mp_score = float(
            mp_analysis["momentum_persistence"]["momentum_persistence"]
        )

    # Dual blend (informational): 50% edge + 30% mom + 20% persist
    dual = None
    try:
        from src.analytics.momentum_persistence_engine import dual_system_blend

        mom_s = float(
            (mp_analysis.get("momentum") or {}).get("momentum_score") or 50
        )
        per_s = float(
            (mp_analysis.get("persistence") or {}).get("effective_persistence")
            or (mp_analysis.get("persistence") or {}).get("persistence")
            or 50
        )
        dual = dual_system_blend(
            existing_edge=float(live_edge),
            momentum_score=mom_s,
            persistence_score=per_s,
        )
    except Exception:
        dual = None

    result = evaluate_no_trade(
        contract_type=ct,
        family=family,
        pattern_clarity=pattern_clarity,
        pattern_strength=pattern_strength,
        hpp=hpp,
        hpp_velocity=hpp_velocity,
        entropy_stability=stability,
        confidence=conf_100,
        signal_confidence=signal_confidence,
        p_win=p_win,
        reward=reward,
        risk=1.0,
        regime_raw=str(regime_raw),
        realtime_pattern_strength=rt_ps,
        peak_hpp=peak_hpp,
        cold_start=cold_start,
        momentum_persistence=mp_score,
        min_confidence=70.0,
    )
    result["_hpp"] = round(hpp, 1)
    result["_hpp_velocity"] = round(hpp_velocity, 2)
    result["_entropy_stability"] = round(stability, 1)
    result["_mp_score"] = mp_score
    result["dual_system"] = dual
    result["momentum_persistence_detail"] = mp_analysis or None

    # Meta-validator: all key metrics must agree (blocks decaying edge)
    try:
        from src.analytics.meta_validator import meta_validate

        meta = meta_validate(
            contract_type=ct,
            pattern_strength=pattern_strength,
            pattern_clarity=pattern_clarity,
            hpp=hpp,
            hpp_velocity=hpp_velocity,
            confidence=conf_100,
            p_win=float(result.get("p_win") or p_win or 0.5),
            reward=reward,
            regime_raw=str(regime_raw),
            realtime_pattern_strength=rt_ps,
            family=family,
            momentum_persistence=mp_score,
            rf_score=(
                float(rf_analysis["rf_score"])
                if rf_analysis.get("rf_score") is not None
                else None
            ),
            vol_tradeable=rf_analysis.get("vol_tradeable")
            if rf_analysis
            else None,
            mp_analysis=mp_analysis or None,
            cold_start=cold_start,
        )
        result["meta_validator"] = meta
        # Meta BLOCK overrides allow unless cold-start already soft-allowed with EV
        if not meta.get("allow"):
            if result.get("allow") and not cold_start:
                result["allow"] = False
                result["status"] = "REJECTED"
                result["reason"] = meta.get("reason") or "Meta-validator BLOCKED"
                result["blocks"] = list(result.get("blocks") or []) + [
                    f"Meta: {meta.get('reason')}"
                ]
                disp = dict(result.get("display") or {})
                disp["status"] = "REJECTED"
                disp["reason"] = result["reason"]
                result["display"] = disp
            elif not result.get("allow"):
                # Keep rejected; enrich reason if velocity decay
                if "decaying" in str(meta.get("reason") or "").lower():
                    result["reason"] = meta["reason"]
    except Exception as e:
        result["meta_validator"] = {"status": "ERROR", "allow": True, "error": str(e)}

    return result


def _copilot_blurb(
    *,
    symbol: str,
    contract_type: str,
    snap: Dict[str, Any],
    ps: Dict[str, Any],
    live: Dict[str, Any],
    hist: Dict[str, Any],
    clarity: Dict[str, Any],
    recommendation: str,
    n_samples: int,
) -> str:
    since = snap.get("ticks_since") or {}
    cold = (
        ((snap.get("heatmap") or {}).get("windows") or {}).get("100", {}).get("cold")
        or []
    )
    lines = []
    if cold:
        d = cold[0]
        tsl = since.get(d)
        if tsl is not None:
            lines.append(
                f"Digit {d} is cold — last seen ~{tsl} ticks ago (100-tick window)."
            )
    n = (hist.get("metrics") or {}).get("n") or n_samples
    bayes = (hist.get("metrics") or {}).get("bayes_wr")
    if n and bayes is not None:
        lines.append(
            f"Similar {contract_type or 'setups'} bayesian WR ≈ {bayes:.0%} over {n} samples."
        )
    lines.append(
        f"Pattern strength {ps.get('pattern_strength')} ({ps.get('class')}); "
        f"clarity {clarity.get('pattern_clarity')} ({clarity.get('class')}); "
        f"live edge {live.get('live_edge')} → {recommendation}."
    )
    if n_samples < MIN_SAMPLE_SIZE:
        lines.append(
            f"Need ≥{MIN_SAMPLE_SIZE} samples for auto-exec (have {n_samples})."
        )
    lines.append(
        f"Suggested risk: 0.5–1.5% of account (status {live.get('risk', 'MEDIUM')})."
    )
    return " ".join(lines)
