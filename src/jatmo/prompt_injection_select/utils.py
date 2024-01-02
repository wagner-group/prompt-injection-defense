from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import dacite


@dataclass(frozen=False)
class ConfigSpec:
    """Configuration spec"""

    path: str = ""
    task_list: List[str] = field(default_factory=list, hash=False)
    orig_file: str = ""
    count: int = 25
    max_injections: int = 1000

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> ConfigSpec:
        return dacite.from_dict(ConfigSpec, d)
