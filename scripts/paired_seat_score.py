"""Score paired seeds after excluding unchanged seat winner relationships."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def score_pairs(report):
    grouped = defaultdict(dict)
    for record in report["records"]:
        seat = record.get("seat", record.get("candidate_seat"))
        if seat not in (0, 1) or seat in grouped[record["pair"]]:
            raise ValueError("Each seed must have exactly one record per candidate seat")
        grouped[record["pair"]][seat] = record["score"]
    counts = dict(a_two_wins=0, a_win_draw=0, b_two_wins=0, b_win_draw=0,
                  first_wins_both=0, second_wins_both=0, two_draws=0, incomplete=0)
    categories = {
        (1., 1.): "a_two_wins", (1., .5): "a_win_draw", (.5, 1.): "a_win_draw",
        (0., 0.): "b_two_wins", (0., .5): "b_win_draw", (.5, 0.): "b_win_draw",
        (1., 0.): "first_wins_both", (0., 1.): "second_wins_both", (.5, .5): "two_draws",
    }
    for records in grouped.values():
        if len(records) != 2 or any(score is None for score in records.values()):
            counts["incomplete"] += 1
            continue
        key = (records[0], records[1])
        if key not in categories:
            raise ValueError(f"Invalid pair outcomes: {key}")
        counts[categories[key]] += 1
    a = counts["a_two_wins"] + counts["a_win_draw"]
    b = counts["b_two_wins"] + counts["b_win_draw"]
    excluded = counts["first_wins_both"] + counts["second_wins_both"] + counts["two_draws"]
    if counts["incomplete"] == 0 and all(k in report for k in ("wins", "draws", "losses")):
        seat_decisive = counts["first_wins_both"] + counts["second_wins_both"]
        reconstructed = (
            2 * counts["a_two_wins"] + counts["a_win_draw"] + seat_decisive,
            counts["a_win_draw"] + counts["b_win_draw"] + 2 * counts["two_draws"],
            2 * counts["b_two_wins"] + counts["b_win_draw"] + seat_decisive,
        )
        if reconstructed != (report["wins"], report["draws"], report["losses"]):
            raise ValueError("Pair categories disagree with original W/D/L totals")
    return {
        "a_model": report["candidate"], "b_model": report["opponent"],
        "simulations": report["simulations"], "seed": report["seed"],
        "total_pairs": len(grouped), "a_points": a, "b_points": b,
        "informative_pairs": a + b, "excluded_pairs": excluded,
        "a_conditional_score_rate": a / (a + b) if a + b else None,
        "categories": counts,
        "interpretation": "One point per retained seed pair. Conditional on retained pairs; not the all-game win/score rate.",
    }
