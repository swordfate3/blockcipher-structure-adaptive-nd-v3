"""Worst-case static counts; never select candidates."""
from typing import cast

from cipher_lab_plugins.integral_search.config import SearchRecipe


def estimate(recipe: SearchRecipe) -> dict[str, object]:
    groups = recipe.groups.discovery + recipe.groups.screen + recipe.groups.confirm
    search_cells: list[dict[str, object]] = []
    for tier, active, patterns in (("discovery", recipe.discovery_active, 2),
                                   ("expanded", recipe.expanded_active, 1)):
        for rounds in recipe.rounds:
            for pattern in range(patterns):
                search_cells.append({
                    "rounds": rounds, "tier": tier,
                    "pattern_index": pattern if tier == "discovery" else None,
                    "conditional": tier == "expanded",
                    "active_count": active, "points_per_group": 1 << active,
                    "planned_real_groups": groups, "planned_null_groups": groups,
                    "planned_real_blocks": groups * (1 << active),
                    "planned_null_words": groups * (1 << active),
                })
    words = sum(cast(int, cell["planned_real_blocks"]) for cell in search_cells)
    return {
        "basis": ("Worst-case full D/S/C on both sides: two discovery patterns per round, "
                  "one conditional expanded pattern per round; early skips are not a "
                  "budget guarantee. Selection is not performed by check."),
        "units": {
            "groups": "one complete cube on one side (real or null)",
            "real_blocks": "one 64-bit plaintext encryption, not round operations",
            "null_words": "one independent 64-bit random-function output per point",
            "optimizer_updates": "one optimizer step; integral uses none",
            "forward_rows": "one model sample row; integral uses none",
        },
        "planned_real_groups": len(search_cells) * groups,
        "planned_null_groups": len(search_cells) * groups,
        "planned_real_blocks": words, "planned_null_words": words,
        "planned_optimizer_updates": 0, "planned_forward_rows": 0,
        "cells": search_cells,
    }
