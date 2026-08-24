"""Oracle-only objects.

This module is intentionally separate from the ordinary domain import surface.
Only the generator, oracle baselines and experimental metric code may import it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict


class TruthBundle(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    true_graph_id: str | None
    potential_outcomes: dict[str, tuple[np.ndarray, np.ndarray]]
    true_ate: dict[str, float]
    true_att: dict[str, float]
    true_cate: dict[str, np.ndarray]
    action_values: dict[str, float]
    optimal_action: str
    admissible_actions: tuple[str, ...] = ("a0", "a1", "a2")
    hidden_factor_present: bool = False
    metadata: dict[str, Any] = {}
