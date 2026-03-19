"""Utilities for tag/nl/xml prompt generation."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple
from xml.sax.saxutils import escape as _xml_escape


def split_tags(tag_text: str) -> List[str]:
    if not tag_text:
        return []
    # Common tag files are comma-separated; tolerate newlines.
    raw = tag_text.replace("\r", "\n").replace("\t", " ")
    parts = []
    for chunk in raw.split("\n"):
        for p in chunk.split(","):
            t = (p or "").strip()
            if t:
                parts.append(t)
    return parts


def pick_artist_characters(tags: List[str]) -> Tuple[str, str, str, List[str]]:
    tags = [t for t in (tags or []) if isinstance(t, str) and t.strip()]
    artist = tags[0] if len(tags) >= 1 else ""
    c1 = tags[1] if len(tags) >= 2 else ""
    c2 = tags[2] if len(tags) >= 3 else ""
    rest = tags[3:] if len(tags) > 3 else []
    return artist, c1, c2, rest


def _normalize_tag_index(value, default: int) -> int:
    try:
        idx = int(value)
    except Exception:
        idx = default
    return max(1, idx)


def _normalize_field_path(value: str) -> str:
    parts = []
    for raw in str(value or "").replace("\\", "/").split("/"):
        token = raw.strip()
        if not token:
            continue
        token = re.sub(r"[^A-Za-z0-9_.-]", "_", token)
        token = token.strip("._-")
        if token:
            parts.append(token)
    return "/".join(parts)


def _collect_xml_mappings(tags: List[str], xml_config: Dict | None) -> List[Tuple[str, str]]:
    tags = [t for t in (tags or []) if isinstance(t, str) and t.strip()]
    cfg = xml_config or {}

    standard_rows = [
        (
            _normalize_field_path(cfg.get("artist_field_path") or "artist"),
            _normalize_tag_index(cfg.get("artist_tag_index", 1), 1),
        ),
        (
            _normalize_field_path(cfg.get("character_1_field_path") or "character_1"),
            _normalize_tag_index(cfg.get("character_1_tag_index", 2), 2),
        ),
        (
            _normalize_field_path(cfg.get("character_2_field_path") or "character_2"),
            _normalize_tag_index(cfg.get("character_2_tag_index", 3), 3),
        ),
    ]

    custom_rows = []
    for item in cfg.get("custom_mappings", []) or []:
        if not isinstance(item, dict):
            continue
        field_path = _normalize_field_path(item.get("field_path", ""))
        tag_index = _normalize_tag_index(item.get("tag_index", 1), 1)
        custom_rows.append((field_path, tag_index))

    resolved: List[Tuple[str, str]] = []
    seen = set()
    for field_path, tag_index in standard_rows + custom_rows:
        if not field_path or field_path in seen:
            continue
        tag_pos = tag_index - 1
        if 0 <= tag_pos < len(tags):
            value = tags[tag_pos].strip()
            if value:
                resolved.append((field_path, value))
                seen.add(field_path)
    return resolved


def _build_xml_tree(mappings: List[Tuple[str, str]]) -> Dict[str, Dict]:
    root: Dict[str, Dict] = {}
    for field_path, value in mappings:
        node = root
        parts = [p for p in field_path.split("/") if p]
        if not parts:
            continue
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        leaf = parts[-1]
        node.setdefault(leaf, {})
        node[leaf]["__value__"] = value
    return root


def _render_xml_tree(tree: Dict[str, Dict], indent: int = 0) -> List[str]:
    lines: List[str] = []
    pad = "  " * indent
    for tag_name, payload in tree.items():
        if not isinstance(payload, dict):
            lines.append(f"{pad}<{tag_name}>{_xml_escape(str(payload))}</{tag_name}>")
            continue
        children = {k: v for k, v in payload.items() if k != "__value__"}
        value = payload.get("__value__", "")
        if children:
            lines.append(f"{pad}<{tag_name}>")
            if value:
                lines.append(f"{pad}  {_xml_escape(str(value))}")
            lines.extend(_render_xml_tree(children, indent + 1))
            lines.append(f"{pad}</{tag_name}>")
        else:
            lines.append(f"{pad}<{tag_name}>{_xml_escape(str(value))}</{tag_name}>")
    return lines


def build_nl_prompt(tag_text: str) -> str:
    tags = split_tags(tag_text)
    artist, c1, c2, rest = pick_artist_characters(tags)
    parts = []
    if artist:
        parts.append(f"artist: {artist}")
    chars = [x for x in (c1, c2) if x]
    if chars:
        parts.append(f"characters: {', '.join(chars)}")
    if rest:
        parts.append(f"tags: {', '.join(rest)}")
    if not parts and tags:
        return f"tags: {', '.join(tags)}"
    return "; ".join(parts)


def build_xml_fragment(tag_text: str, xml_config: Dict | None = None) -> str:
    tags = split_tags(tag_text)
    mappings = _collect_xml_mappings(tags, xml_config)
    if not mappings:
        return ""
    tree = _build_xml_tree(mappings)
    return "\n".join(_render_xml_tree(tree))
