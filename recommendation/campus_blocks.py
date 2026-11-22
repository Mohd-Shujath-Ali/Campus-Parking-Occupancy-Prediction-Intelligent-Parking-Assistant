"""
Campus block directory: maps any block a user can belong to onto the reference destination
whose walking distances are in configs/zones.yaml.

A student or faculty member whose home block is adjacent to a reference block (for example
Block 55 next to Block 57) is ranked by distances from that reference block, so they get the
parking grounds nearest to them.
"""

from functools import lru_cache
from typing import Dict, List, Optional

import yaml


def _norm(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_").replace("-", "_")


class CampusBlocks:
    def __init__(self, blocks_path: str = "configs/campus_blocks.yaml", zones_path: str = "configs/zones.yaml"):
        with open(zones_path, "r") as f:
            zones = yaml.safe_load(f)["zones"]
        self.anchors = sorted({d for z in zones for d in z.get("walking_distances_meters", {})})
        with open(blocks_path, "r") as f:
            raw = yaml.safe_load(f).get("blocks", [])

        anchor_norm = {_norm(a): a for a in self.anchors}
        self.blocks: List[Dict[str, str]] = []
        seen = set()
        for b in raw:
            anchor = anchor_norm.get(_norm(b["anchor"]))
            if anchor is None:
                raise ValueError(f"campus_blocks.yaml: block '{b['id']}' has unknown anchor '{b['anchor']}'. "
                                 f"Valid anchors: {self.anchors}")
            if _norm(b["id"]) in seen:
                raise ValueError(f"campus_blocks.yaml: duplicate block id '{b['id']}'")
            seen.add(_norm(b["id"]))
            self.blocks.append({"id": b["id"], "name": b.get("name", b["id"]), "anchor": anchor,
                                "group": b.get("group", "Other"), "is_reference": _norm(b["id"]) == _norm(anchor)})
        # Every anchor is a valid block even if the YAML forgot to list it.
        for a in self.anchors:
            if _norm(a) not in seen:
                self.blocks.append({"id": a, "name": a.replace("_", " "), "anchor": a,
                                    "group": "Other", "is_reference": True})
        self._by_norm = {_norm(b["id"]): b for b in self.blocks}

    def get(self, block_id: Optional[str]) -> Optional[Dict[str, str]]:
        return self._by_norm.get(_norm(block_id)) if block_id else None

    def resolve_anchor(self, block_id: Optional[str]) -> Optional[str]:
        b = self.get(block_id)
        return b["anchor"] if b else None

    def canonical_id(self, block_id: Optional[str]) -> Optional[str]:
        b = self.get(block_id)
        return b["id"] if b else None

    def all(self) -> List[Dict[str, str]]:
        return list(self.blocks)


@lru_cache(maxsize=1)
def get_campus_blocks() -> CampusBlocks:
    return CampusBlocks()
