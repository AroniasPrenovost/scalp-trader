"""
Core Learnings System - Persistent pattern avoidance and calibration
Tracks what works, what doesn't, and enforces hard rules to prevent repeated mistakes.
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

LEARNINGS_DIR = "learnings"


def ensure_learnings_directory():
    """Ensure the learnings directory exists."""
    if not os.path.exists(LEARNINGS_DIR):
        os.makedirs(LEARNINGS_DIR)


def get_learnings_path(symbol: str) -> str:
    """Get the file path for a symbol's core learnings."""
    ensure_learnings_directory()
    return os.path.join(LEARNINGS_DIR, f"{symbol}_core_rules.json")


def initialize_core_learnings(symbol: str) -> Dict:
    """Initialize a new core learnings structure for a symbol."""
    return {
        "version": 1,
        "symbol": symbol,
        "last_updated": datetime.now().isoformat(),
        "hard_rules": [],
        "calibrations": {
            "support_levels": {},
            "resistance_levels": {},
            "confidence_thresholds": {
                "high": {"min_win_rate": 0.70, "current": 0.0, "status": "insufficient_data"}
            }
        },
        "pattern_blacklist": [],
        "loss_streaks": {
            "current_streak": 0,
            "max_streak": 0,
            "last_loss_date": None
        },
        "metadata": {
            "total_trades_analyzed": 0,
            "rules_triggered": 0,
            "patterns_blacklisted": 0
        }
    }


def load_core_learnings(symbol: str) -> Dict:
    """Load core learnings for a symbol, create new if doesn't exist."""
    learnings_path = get_learnings_path(symbol)

    if not os.path.exists(learnings_path):
        return initialize_core_learnings(symbol)

    try:
        with open(learnings_path, 'r') as f:
            learnings = json.load(f)
            # Ensure all required keys exist (for backward compatibility)
            default_learnings = initialize_core_learnings(symbol)
            for key in default_learnings:
                if key not in learnings:
                    learnings[key] = default_learnings[key]
            return learnings
    except Exception as e:
        print(f"⚠️  Error loading core learnings: {e}")
        return initialize_core_learnings(symbol)


def save_core_learnings(symbol: str, learnings: Dict) -> bool:
    """Save core learnings to file."""
    learnings_path = get_learnings_path(symbol)
    learnings["last_updated"] = datetime.now().isoformat()

    try:
        with open(learnings_path, 'w') as f:
            json.dump(learnings, f, indent=2)
        return True
    except Exception as e:
        print(f"⚠️  Error saving core learnings: {e}")
        return False


def evaluate_hard_rules(
    learnings: Dict,
    current_conditions: Dict
) -> Tuple[bool, Optional[str]]:
    """
    Evaluate hard rules that block trading.

    Args:
        learnings: Core learnings dictionary
        current_conditions: Dict with keys like 'day_of_week', 'volume_ratio',
                          'market_trend', 'confidence_level', etc.

    Returns:
        (should_trade, block_reason) - False if any rule blocks trading
    """
    hard_rules = learnings.get("hard_rules", [])

    for rule in hard_rules:
        rule_type = rule.get("rule")
        condition = rule.get("condition", {})

        # Rule: Never trade on weekend with low volume
        if rule_type == "never_trade_weekend_low_volume":
            if (current_conditions.get("day_of_week") in ["Saturday", "Sunday"] and
                current_conditions.get("volume_ratio", 1.0) < condition.get("volume_threshold", 0.5)):
                return False, f"🚫 HARD RULE: {rule.get('reason', 'Weekend low volume')}"

        # Rule: Reduce position after loss streak
        elif rule_type == "reduce_position_after_loss_streak":
            streak = learnings.get("loss_streaks", {}).get("current_streak", 0)
            if streak >= condition.get("streak_threshold", 3):
                # This doesn't block, but will be used in position sizing
                pass

        # Rule: Never trade in specific market condition combinations
        elif rule_type == "never_trade_condition_combination":
            required_conditions = condition.get("conditions", {})
            all_match = True
            for key, expected_value in required_conditions.items():
                if current_conditions.get(key) != expected_value:
                    all_match = False
                    break
            if all_match:
                return False, f"🚫 HARD RULE: {rule.get('reason', 'Condition combination blacklisted')}"

        # Rule: Never trade below minimum confidence after calibration
        elif rule_type == "enforce_minimum_confidence":
            required_confidence = condition.get("minimum_confidence", "high")
            current_confidence = current_conditions.get("confidence_level", "low")
            confidence_order = {"low": 0, "medium": 1, "high": 2}
            if confidence_order.get(current_confidence, 0) < confidence_order.get(required_confidence, 2):
                return False, f"🚫 HARD RULE: {rule.get('reason', 'Confidence too low')}"

    return True, None


def check_pattern_blacklist(
    learnings: Dict,
    pattern_description: str,
    market_trend: str,
    confidence_level: str
) -> Tuple[bool, Optional[str]]:
    """
    Check if a pattern or condition combination is blacklisted.

    Returns:
        (should_trade, block_reason)
    """
    blacklist = learnings.get("pattern_blacklist", [])

    for entry in blacklist:
        pattern = entry.get("pattern", "")
        blacklisted_trend = entry.get("market_trend")
        blacklisted_confidence = entry.get("confidence_level")

        # Check for exact pattern match
        if pattern and pattern.lower() in pattern_description.lower():
            if entry.get("action") == "auto_skip":
                reason = entry.get("reason", f"Pattern blacklisted: {entry.get('losses', 0)} losses")
                return False, f"🚫 BLACKLISTED PATTERN: {reason}"

        # Check for market condition + confidence combination
        if blacklisted_trend and blacklisted_confidence:
            if (market_trend == blacklisted_trend and
                confidence_level == blacklisted_confidence):
                reason = entry.get("reason", "Condition combination has poor track record")
                return False, f"🚫 BLACKLISTED COMBO: {reason}"

    return True, None


def apply_calibrations(
    learnings: Dict,
    analysis: Dict,
    market_trend: str
) -> Dict:
    """
    Apply calibrations to support/resistance levels and other analysis values.

    Args:
        learnings: Core learnings dictionary
        analysis: AI analysis result
        market_trend: Current market trend (bullish/bearish/sideways/ranging)

    Returns:
        Modified analysis with calibrations applied
    """
    calibrations = learnings.get("calibrations", {})

    # Apply support level adjustments
    support_cal = calibrations.get("support_levels", {})
    if market_trend in support_cal:
        adjustment = support_cal[market_trend].get("adjustment", 0)
        if "major_support" in analysis:
            original = analysis["major_support"]
            analysis["major_support"] = original * (1 + adjustment)
            print(f"📊 Applied support calibration ({adjustment:+.1%}) for {market_trend} market")
        if "minor_support" in analysis:
            analysis["minor_support"] = analysis["minor_support"] * (1 + adjustment)

    # Apply resistance level adjustments
    resistance_cal = calibrations.get("resistance_levels", {})
    if market_trend in resistance_cal:
        adjustment = resistance_cal[market_trend].get("adjustment", 0)
        if "major_resistance" in analysis:
            original = analysis["major_resistance"]
            analysis["major_resistance"] = original * (1 + adjustment)
            print(f"📊 Applied resistance calibration ({adjustment:+.1%}) for {market_trend} market")
        if "minor_resistance" in analysis:
            analysis["minor_resistance"] = analysis["minor_resistance"] * (1 + adjustment)

    return analysis


def get_position_size_multiplier(learnings: Dict) -> float:
    """
    Get position size multiplier based on loss streaks.

    Returns:
        Multiplier (0.5 = half position, 1.0 = full position)
    """
    loss_streaks = learnings.get("loss_streaks", {})
    current_streak = loss_streaks.get("current_streak", 0)

    if current_streak >= 5:
        return 0.25  # Quarter position after 5 losses
    elif current_streak >= 3:
        return 0.5   # Half position after 3 losses
    else:
        return 1.0   # Full position


def update_learnings_from_trade(
    symbol: str,
    trade_outcome: Dict,
    transactions_history: List[Dict]
) -> Dict:
    """
    Update core learnings based on completed trade outcome.

    Args:
        symbol: Trading symbol
        trade_outcome: Dict with keys like 'profit', 'exit_trigger', 'confidence_level', etc.
        transactions_history: Full transaction history for pattern analysis

    Returns:
        Updated learnings dictionary
    """
    learnings = load_core_learnings(symbol)

    # Update metadata
    learnings["metadata"]["total_trades_analyzed"] += 1

    # Update loss streak
    is_loss = trade_outcome.get("profit", 0) < 0
    loss_streaks = learnings.get("loss_streaks", {})

    if is_loss:
        loss_streaks["current_streak"] = loss_streaks.get("current_streak", 0) + 1
        loss_streaks["last_loss_date"] = datetime.now().isoformat()
        if loss_streaks["current_streak"] > loss_streaks.get("max_streak", 0):
            loss_streaks["max_streak"] = loss_streaks["current_streak"]
    else:
        loss_streaks["current_streak"] = 0  # Reset on win

    learnings["loss_streaks"] = loss_streaks

    # Add hard rule if loss streak is concerning
    if loss_streaks["current_streak"] >= 3:
        existing_rule = any(r.get("rule") == "reduce_position_after_loss_streak"
                           for r in learnings.get("hard_rules", []))
        if not existing_rule:
            learnings["hard_rules"].append({
                "rule": "reduce_position_after_loss_streak",
                "condition": {"streak_threshold": 3},
                "action": "multiply_position_by_0.5",
                "reason": f"Prevent drawdown spirals - {loss_streaks['current_streak']} losses in a row",
                "added": datetime.now().isoformat()
            })
            print(f"⚠️  Added hard rule: Reduce position after loss streak")

    # Analyze patterns to blacklist
    _analyze_and_blacklist_patterns(learnings, transactions_history)

    # Update calibrations based on support/resistance accuracy
    _update_calibrations(learnings, trade_outcome, transactions_history)

    # Save updated learnings
    save_core_learnings(symbol, learnings)

    return learnings


def _analyze_and_blacklist_patterns(learnings: Dict, transactions: List[Dict]):
    """
    Analyze recent transaction history and blacklist failing patterns.
    """
    if len(transactions) < 5:
        return  # Need at least 5 trades to identify patterns

    # Group by market_trend + confidence_level combinations
    combinations = {}
    for t in transactions[-20:]:  # Last 20 trades
        trend = t.get("market_trend", "unknown")
        confidence = t.get("confidence_level", "unknown")
        key = f"{trend}_{confidence}"

        if key not in combinations:
            combinations[key] = {"wins": 0, "losses": 0, "trades": []}

        if t.get("total_profit", 0) > 0:
            combinations[key]["wins"] += 1
        else:
            combinations[key]["losses"] += 1
        combinations[key]["trades"].append(t)

    # Find combinations with 0% win rate and 3+ attempts
    blacklist = learnings.get("pattern_blacklist", [])

    for combo_key, stats in combinations.items():
        total = stats["wins"] + stats["losses"]
        if total >= 3 and stats["wins"] == 0:
            # This combo has never won after 3+ attempts
            trend, confidence = combo_key.split("_")

            # Check if already blacklisted
            already_blacklisted = any(
                p.get("market_trend") == trend and p.get("confidence_level") == confidence
                for p in blacklist
            )

            if not already_blacklisted:
                blacklist.append({
                    "pattern": combo_key,
                    "market_trend": trend,
                    "confidence_level": confidence,
                    "losses": stats["losses"],
                    "wins": stats["wins"],
                    "action": "auto_skip",
                    "reason": f"0% win rate on {total} attempts ({trend} + {confidence})",
                    "added": datetime.now().isoformat()
                })
                learnings["metadata"]["patterns_blacklisted"] += 1
                print(f"🚫 Pattern blacklisted: {trend} + {confidence} (0/{total} wins)")

    learnings["pattern_blacklist"] = blacklist


def _update_calibrations(learnings: Dict, trade_outcome: Dict, transactions: List[Dict]):
    """
    Update calibrations based on whether support/resistance levels held.
    """
    if len(transactions) < 3:
        return

    calibrations = learnings.get("calibrations", {})

    # Analyze support level accuracy by market trend
    support_accuracy = {}
    for t in transactions[-15:]:  # Last 15 trades
        trend = t.get("market_trend", "unknown")
        if trend not in support_accuracy:
            support_accuracy[trend] = {"held": 0, "broke": 0}

        # Check if price broke below stop loss (support broke)
        if t.get("exit_trigger") == "stop_loss":
            support_accuracy[trend]["broke"] += 1
        else:
            support_accuracy[trend]["held"] += 1

    # Update calibrations if support is consistently breaking
    support_cal = calibrations.get("support_levels", {})
    for trend, stats in support_accuracy.items():
        total = stats["held"] + stats["broke"]
        if total >= 5:
            break_rate = stats["broke"] / total

            # If support breaks >40% of the time, adjust downward
            if break_rate > 0.4:
                adjustment = -0.05 * (break_rate / 0.4)  # Scale adjustment
                if trend not in support_cal:
                    support_cal[trend] = {}
                support_cal[trend]["adjustment"] = adjustment
                support_cal[trend]["reason"] = f"Support broke {break_rate:.0%} of the time"
                support_cal[trend]["updated"] = datetime.now().isoformat()
                print(f"📊 Updated support calibration for {trend}: {adjustment:+.1%}")

    calibrations["support_levels"] = support_cal
    learnings["calibrations"] = calibrations


def format_learnings_for_display(learnings: Dict) -> str:
    """Format learnings for console display."""
    output = []
    output.append("\n" + "="*60)
    output.append("📚 CORE LEARNINGS - PERSISTENT RULES")
    output.append("="*60)

    # Hard rules
    hard_rules = learnings.get("hard_rules", [])
    if hard_rules:
        output.append("\n🚫 HARD RULES (Auto-enforced):")
        for i, rule in enumerate(hard_rules, 1):
            output.append(f"  {i}. {rule.get('reason', 'No reason')}")
    else:
        output.append("\n✓ No hard rules active (no repeated mistakes detected)")

    # Pattern blacklist
    blacklist = learnings.get("pattern_blacklist", [])
    if blacklist:
        output.append("\n⛔ BLACKLISTED PATTERNS:")
        for i, pattern in enumerate(blacklist, 1):
            wins = pattern.get("wins", 0)
            losses = pattern.get("losses", 0)
            output.append(f"  {i}. {pattern.get('pattern', 'Unknown')}: {wins}W-{losses}L - {pattern.get('reason', '')}")

    # Calibrations
    calibrations = learnings.get("calibrations", {})
    support_cal = calibrations.get("support_levels", {})
    if support_cal:
        output.append("\n📊 ACTIVE CALIBRATIONS:")
        for trend, cal in support_cal.items():
            adj = cal.get("adjustment", 0)
            output.append(f"  • {trend.upper()} support: {adj:+.1%} ({cal.get('reason', '')})")

    # Loss streak warning
    loss_streaks = learnings.get("loss_streaks", {})
    current_streak = loss_streaks.get("current_streak", 0)
    if current_streak > 0:
        output.append(f"\n⚠️  CURRENT LOSS STREAK: {current_streak} trades")
        multiplier = get_position_size_multiplier(learnings)
        if multiplier < 1.0:
            output.append(f"  → Position size reduced to {multiplier:.0%}")

    # Metadata
    metadata = learnings.get("metadata", {})
    output.append(f"\n📈 Stats: {metadata.get('total_trades_analyzed', 0)} trades analyzed, " +
                 f"{metadata.get('patterns_blacklisted', 0)} patterns blacklisted")

    output.append("="*60 + "\n")
    return "\n".join(output)


def format_learnings_for_llm(learnings: Dict) -> str:
    """Format learnings for inclusion in LLM prompt."""
    output = []
    output.append("\n" + "="*60)
    output.append("CORE LEARNINGS - MANDATORY RULES AND CALIBRATIONS")
    output.append("="*60)
    output.append("\nThese are HARD RULES derived from repeated failures. Apply them STRICTLY.\n")

    # Hard rules
    hard_rules = learnings.get("hard_rules", [])
    if hard_rules:
        output.append("🚫 HARD RULES (NEVER violate these):")
        for rule in hard_rules:
            output.append(f"  - {rule.get('reason', 'Unknown rule')}")

    # Pattern blacklist
    blacklist = learnings.get("pattern_blacklist", [])
    if blacklist:
        output.append("\n⛔ BLACKLISTED PATTERNS (Auto-reject if detected):")
        for pattern in blacklist:
            output.append(f"  - {pattern.get('pattern', 'Unknown')}: {pattern.get('reason', '')}")

    # Calibrations
    calibrations = learnings.get("calibrations", {})
    support_cal = calibrations.get("support_levels", {})
    resistance_cal = calibrations.get("resistance_levels", {})

    if support_cal or resistance_cal:
        output.append("\n📊 MANDATORY CALIBRATIONS (Adjust your analysis):")
        for trend, cal in support_cal.items():
            output.append(f"  - {trend.upper()} support levels: Adjust {cal.get('adjustment', 0):+.1%} " +
                         f"({cal.get('reason', '')})")
        for trend, cal in resistance_cal.items():
            output.append(f"  - {trend.upper()} resistance levels: Adjust {cal.get('adjustment', 0):+.1%} " +
                         f"({cal.get('reason', '')})")

    # Loss streak warning
    loss_streaks = learnings.get("loss_streaks", {})
    current_streak = loss_streaks.get("current_streak", 0)
    if current_streak >= 3:
        output.append(f"\n⚠️  CRITICAL WARNING: Currently on {current_streak}-trade LOSS STREAK")
        output.append("  → ONLY trade EXCEPTIONAL setups")
        output.append("  → Increase confidence threshold significantly")
        output.append("  → Position size will be automatically reduced")

    output.append("\n" + "="*60 + "\n")
    return "\n".join(output)
