from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union
from datetime import datetime, timezone


def write_stage_meta(meta_path: Union[str, Path], meta: Dict[str, Any]) -> None:
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(meta)
    meta.setdefault("schema_version", 1)
    meta.setdefault("utc_time", datetime.now(timezone.utc).isoformat())
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
