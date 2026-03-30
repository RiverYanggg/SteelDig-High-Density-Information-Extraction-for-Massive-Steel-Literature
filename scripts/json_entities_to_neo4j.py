#!/usr/bin/env python3
"""
Import paper_entity_schema JSON from datasets/output_text into Neo4j.

Readability (nodes / edges / PNG export):
- Every node has **display** as the main caption (not raw kind).
- Labels: **JsonObject** / **JsonArray** / **JsonScalar** (see neo4j-browser-graph-style.grass).
- Top-level semantic relationships: HAS_METADATA, HAS_PROCESSING, ...
- Nested: **HAS_FIELD** with **label** (English) and **field** (original key).
- Arrays: **AT_INDEX** with **label** like [0], [1].

Neo4j Browser: Settings -> Graph style -> Import
  datasets/neo4j_runtime/neo4j-browser-graph-style.grass

Usage:
  .venv/bin/python scripts/json_entities_to_neo4j.py --clear
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(Path.cwd() / ".env")
except ImportError:
    pass

try:
    from neo4j import GraphDatabase
except ImportError as e:
    raise SystemExit("Install neo4j driver: pip install neo4j") from e


DEFAULT_TEXT_ENTITIES_DIR = PROJECT_ROOT / "datasets" / "output_text"

# Top-level JSON keys -> relationship types (whitelist)
TOP_REL_TYPE: Dict[str, str] = {
    "metadata": "HAS_METADATA",
    "processing": "HAS_PROCESSING",
    "structure": "HAS_STRUCTURE",
    "interface": "HAS_INTERFACE",
    "properties": "HAS_PROPERTIES",
    "performance": "HAS_PERFORMANCE",
    "characterization_methods": "HAS_CHARACTERIZATION",
    "computational_details": "HAS_COMPUTATIONAL",
    "unmapped_findings": "HAS_UNMAPPED",
}

# Optional: friendlier English for known keys (display + edge label). Unknown keys -> Title Case words.
KEY_LABEL_EN: Dict[str, str] = {
    "metadata": "Metadata",
    "processing": "Processing",
    "structure": "Microstructure",
    "interface": "Interface",
    "properties": "Properties",
    "performance": "Performance",
    "characterization_methods": "Characterization",
    "computational_details": "Computational",
    "unmapped_findings": "Unmapped findings",
    "doi": "DOI",
    "title": "Title",
    "authors": "Authors",
    "publication_year": "Year",
    "journal": "Journal",
    "keywords": "Keywords",
    "research_type": "Research type",
    "alloy_system": "Alloy system",
    "base_element": "Base element",
    "alloying_elements": "Alloying elements",
    "nominal_composition": "Nominal composition",
    "element": "Element",
    "weight_percent": "Weight %",
    "atomic_percent": "Atomic %",
    "synthesis_methods": "Synthesis",
    "process_sequence": "Process sequence",
    "post_processing": "Post-processing",
    "parameters": "Parameters",
    "temperature": "Temperature",
    "atmosphere": "Atmosphere",
    "cooling_rate": "Cooling rate",
    "others": "Other",
    "sequence": "Step",
    "type": "Type",
    "method": "Method",
    "unit": "Unit",
    "duration": "Duration",
    "cooling_medium": "Cooling medium",
    "reduction_ratio": "Reduction",
    "overall_structure": "Overall structure",
    "number_of_phases": "Number of phases",
    "microstructure_counts": "Microstructure count",
    "microstructure_list": "Microstructure list",
    "uuid": "UUID",
    "related_sequence": "Related process step",
    "phases_present": "Phases present",
    "phase_name": "Phase",
    "crystal_structure": "Crystal structure",
    "volume_fraction": "Volume fraction",
    "morphology": "Morphology",
    "grain_size": "Grain size",
    "texture": "Texture",
    "defects": "Defects",
    "grain_structure": "Grain structure",
    "average_grain_size": "Average grain size",
    "precipitates": "Precipitates",
    "distribution": "Distribution",
    "coherency": "Coherency",
    "phases": "Phase pairs",
    "phase_1_name": "Phase 1",
    "phase_2_name": "Phase 2",
    "coherence": "Coherence",
    "defect_interaction": "Defect interaction",
    "interaction_type": "Interaction type",
    "quality_description": "Description",
    "phase_evolution": "Phase evolution",
    "mechanical": "Mechanical",
    "physical": "Physical",
    "chemical": "Chemical",
    "radiation_properties": "Radiation",
    "tensile_properties": "Tensile",
    "yield_strength": "Yield strength",
    "ultimate_tensile_strength": "UTS",
    "elongation": "Elongation",
    "youngs_modulus": "Young's modulus",
    "hardness": "Hardness",
    "value": "Value",
    "direction": "Direction",
}


def _key_label(k: str) -> str:
    if k in KEY_LABEL_EN:
        return KEY_LABEL_EN[k]
    return k.replace("_", " ").title()


def _truncate(s: str, max_len: int = 42) -> str:
    s = s.replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "..."


def _file_id_from_path(path: Path) -> str:
    rel = str(path.relative_to(PROJECT_ROOT))
    return hashlib.sha256(rel.encode("utf-8")).hexdigest()[:16]


def _escape_seg(segment: str) -> str:
    return segment.replace("~", "~0").replace("/", "~1")


def _ptr_join(base: str, segment: Union[str, int]) -> str:
    seg = str(segment)
    if not base or base == "/":
        return "/" + seg
    return base + "/" + seg


def drop_all_constraints(tx) -> None:
    for row in tx.run("SHOW CONSTRAINTS"):
        name = row.get("name")
        if not name:
            continue
        safe = str(name).replace("`", "")
        tx.run(f"DROP CONSTRAINT `{safe}` IF EXISTS")


def clear_graph(tx) -> None:
    tx.run("MATCH (n) DETACH DELETE n")


def ensure_constraints(tx) -> None:
    tx.run(
        "CREATE CONSTRAINT source_file_id IF NOT EXISTS FOR (f:SourceFile) REQUIRE f.file_id IS UNIQUE"
    )
    tx.run(
        "CREATE CONSTRAINT json_entity_uid IF NOT EXISTS FOR (n:JsonEntity) REQUIRE n.uid IS UNIQUE"
    )


def list_json_files(root: Path) -> List[Path]:
    skip = {"package-lock", "package", "tsconfig"}
    out: List[Path] = []
    for p in sorted(root.glob("*.json")):
        if p.stem.lower() in skip:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and "metadata" in data and "processing" in data:
            out.append(p)
    return out


def ingest_json_tree(session, path: Path) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Root value must be a JSON object: {path}")

    file_id = _file_id_from_path(path)
    rel_path = str(path.relative_to(PROJECT_ROOT))

    def uid(ptr: str) -> str:
        return f"{file_id}#{ptr}"

    def write(tx):
        disp_file = path.stem if len(path.stem) <= 48 else path.stem[:47] + "..."
        tx.run(
            """
            MERGE (f:SourceFile {file_id: $file_id})
            SET f.file_name = $file_name,
                f.stem = $stem,
                f.relative_path = $rel_path,
                f.schema_hint = 'paper_entity_schema',
                f.display = $display
            """,
            file_id=file_id,
            file_name=path.name,
            stem=path.stem,
            rel_path=rel_path,
            display=f"File: {disp_file}",
        )

        def merge_node(
            label: str,
            ptr: str,
            *,
            display: str,
            kind: str,
            length: Optional[int] = None,
            is_null: Optional[bool] = None,
            str_val: Optional[str] = None,
        ) -> None:
            u = uid(ptr)
            if label == "JsonObject":
                tx.run(
                    """
                    MERGE (n:JsonEntity:JsonObject {uid: $uid})
                    SET n.display = $display, n.path = $ptr, n.kind = $kind
                    """,
                    uid=u,
                    display=display,
                    ptr=ptr,
                    kind=kind,
                )
            elif label == "JsonArray":
                tx.run(
                    """
                    MERGE (n:JsonEntity:JsonArray {uid: $uid})
                    SET n.display = $display, n.path = $ptr, n.kind = $kind, n.length = $length
                    """,
                    uid=u,
                    display=display,
                    ptr=ptr,
                    kind=kind,
                    length=length,
                )
            else:
                tx.run(
                    """
                    MERGE (n:JsonEntity:JsonScalar {uid: $uid})
                    SET n.display = $display,
                        n.path = $ptr,
                        n.kind = $kind,
                        n.is_null = $is_null,
                        n.value = $str_val
                    """,
                    uid=u,
                    display=display,
                    ptr=ptr,
                    kind=kind,
                    is_null=is_null,
                    str_val=str_val,
                )

        def link_source(child_uid: str, top_key: str, child_label: str) -> None:
            rel = TOP_REL_TYPE.get(top_key, "HAS_TOP")
            allowed = set(TOP_REL_TYPE.values()) | {"HAS_TOP"}
            if rel not in allowed:
                rel = "HAS_TOP"
            tx.run(
                f"""
                MATCH (f:SourceFile {{file_id: $fid}})
                MATCH (c:{child_label} {{uid: $cuid}})
                MERGE (f)-[r:{rel}]->(c)
                """,
                fid=file_id,
                cuid=child_uid,
            )

        def link_field(parent_uid: str, child_uid: str, field: str, label_txt: str) -> None:
            tx.run(
                """
                MATCH (p:JsonEntity {uid: $puid})
                MATCH (c:JsonEntity {uid: $cuid})
                MERGE (p)-[r:HAS_FIELD]->(c)
                SET r.field = $field, r.label = $label_txt
                """,
                puid=parent_uid,
                cuid=child_uid,
                field=field,
                label_txt=label_txt,
            )

        def link_index(parent_uid: str, child_uid: str, index: int) -> None:
            lbl = f"[{index}]"
            tx.run(
                """
                MATCH (p:JsonEntity {uid: $puid})
                MATCH (c:JsonEntity {uid: $cuid})
                MERGE (p)-[r:AT_INDEX]->(c)
                SET r.index = $index, r.label = $lbl
                """,
                puid=parent_uid,
                cuid=child_uid,
                index=index,
                lbl=lbl,
            )

        def visit(
            value: Any,
            ptr: str,
            parent: str,
            *,
            field_key: Optional[str] = None,
            item_index: Optional[int] = None,
        ) -> None:
            c_uid = uid(ptr)
            key_disp = _key_label(field_key) if field_key else ""

            if isinstance(value, dict):
                disp = f"{key_disp} · object" if field_key else "object"
                merge_node("JsonObject", ptr, display=disp, kind="object")
                if parent == "SOURCE":
                    assert field_key is not None
                    link_source(c_uid, field_key, "JsonObject")
                elif field_key is not None:
                    link_field(parent, c_uid, field_key, key_disp)
                else:
                    link_index(parent, c_uid, item_index if item_index is not None else 0)

                for k, v in value.items():
                    seg = _escape_seg(k)
                    visit(v, _ptr_join(ptr, seg), c_uid, field_key=k)

            elif isinstance(value, list):
                n = len(value)
                disp = f"{key_disp} · {n} items" if field_key else f"list · {n} items"
                merge_node("JsonArray", ptr, display=disp, kind="array", length=n)
                if parent == "SOURCE":
                    assert field_key is not None
                    link_source(c_uid, field_key, "JsonArray")
                elif field_key is not None:
                    link_field(parent, c_uid, field_key, key_disp)
                else:
                    link_index(parent, c_uid, item_index if item_index is not None else 0)

                for i, v in enumerate(value):
                    visit(v, _ptr_join(ptr, i), c_uid, item_index=i)

            else:
                is_null = value is None
                if is_null:
                    disp = f"{key_disp}: (null)" if field_key else "(null)"
                    str_val = None
                else:
                    raw_s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                    str_val = raw_s
                    body = "(empty string)" if raw_s == "" else _truncate(raw_s, 48)
                    disp = f"{key_disp}: {body}" if field_key else body
                merge_node(
                    "JsonScalar",
                    ptr,
                    display=disp,
                    kind="scalar",
                    is_null=is_null,
                    str_val=str_val,
                )
                if parent == "SOURCE":
                    assert field_key is not None
                    link_source(c_uid, field_key, "JsonScalar")
                elif field_key is not None:
                    link_field(parent, c_uid, field_key, key_disp)
                else:
                    link_index(parent, c_uid, item_index if item_index is not None else 0)

        for top_key, v in raw.items():
            visit(v, _ptr_join("", _escape_seg(top_key)), "SOURCE", field_key=top_key)

    session.execute_write(write)
    print(f"  OK {path.name}  file_id={file_id}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Import JSON tree into Neo4j (English labels)")
    ap.add_argument("--input", type=Path, default=DEFAULT_TEXT_ENTITIES_DIR)
    ap.add_argument("--clear", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "steeldig2026")

    root: Path = args.input
    if not root.is_dir():
        print(f"Directory not found: {root}", file=sys.stderr)
        return 1

    files = list_json_files(root)
    if not files:
        print(f"No entity JSON files found: {root}", file=sys.stderr)
        return 1

    driver = GraphDatabase.driver(uri, auth=(user, password))
    grass = PROJECT_ROOT / "datasets" / "neo4j_runtime" / "neo4j-browser-graph-style.grass"
    try:
        with driver.session() as session:
            if args.clear:
                session.execute_write(drop_all_constraints)
                session.execute_write(clear_graph)
            session.execute_write(ensure_constraints)
            print(f"Importing {len(files)} file(s) -> {uri}")
            for p in files:
                print("Import", p.name)
                ingest_json_tree(session, p)
        print("Done.")
        print("Neo4j Browser: Settings -> Graph style -> import (captions use `display`):")
        print(f"  {grass}")
        print("Example:")
        print("  MATCH (f:SourceFile)-[r]->(n) RETURN f.display, type(r), n.display LIMIT 30")
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
