#!/usr/bin/env python3
"""Extract a reproducible Homo sapiens BACE1 IC50 dataset from ChEMBL 36."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_CHEMBL_ID = "CHEMBL4822"
TARGET_ORGANISM = "Homo sapiens"
ACTIVITY_TYPE = "IC50"

ACTIVITY_COLUMNS = [
    "activity_id",
    "assay_id",
    "doc_id",
    "molregno",
    "canonical_smiles",
    "standard_inchi_key",
    "standard_value",
    "standard_units",
    "standard_relation",
    "pic50",
    "potential_duplicate",
    "data_validity_comment",
]

OUTPUT_COLUMNS = ACTIVITY_COLUMNS + [
    "confidence_score",
    "assay_type",
    "target_chembl_id",
    "target_name",
    "target_organism",
    "target_type",
]

ACTIVITY_QUERY_TEMPLATE = """
SELECT
    act.activity_id,
    act.assay_id,
    act.doc_id,
    act.molregno,
    cs.canonical_smiles,
    cs.standard_inchi_key,
    act.standard_value,
    act.standard_units,
    act.standard_relation,
    act.pchembl_value AS pic50,
    act.potential_duplicate,
    act.data_validity_comment
FROM activities AS act INDEXED BY fk_act_assay_id
JOIN compound_structures AS cs
    ON cs.molregno = act.molregno
WHERE
    act.assay_id IN ({assay_placeholders})
    AND act.standard_type = ?
    AND act.standard_relation = '='
    AND act.standard_units = 'nM'
    AND act.standard_value > 0
    AND act.pchembl_value IS NOT NULL
    AND cs.canonical_smiles IS NOT NULL
    AND TRIM(cs.canonical_smiles) <> ''
    AND COALESCE(act.potential_duplicate, 0) = 0
    AND act.data_validity_comment IS NULL
ORDER BY act.assay_id, act.activity_id
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract strict Homo sapiens BACE1 IC50 records from ChEMBL 36."
    )
    parser.add_argument("--db", type=Path, default=Path("chembl_36.db"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/bace1_chembl36_strict.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/raw/bace1_chembl36_strict_report.json"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional deterministic row limit used only for a smoke test.",
    )
    return parser.parse_args()


def connect_read_only(db_path: Path) -> sqlite3.Connection:
    resolved = db_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"ChEMBL database not found: {resolved}")

    connection = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def validate_database(connection: sqlite3.Connection) -> dict[str, Any]:
    version_row = connection.execute(
        """
        SELECT name, creation_date
        FROM version
        WHERE name LIKE 'ChEMBL_%'
        ORDER BY creation_date DESC
        LIMIT 1
        """
    ).fetchone()
    target_row = connection.execute(
        """
        SELECT tid, chembl_id, pref_name, organism, target_type
        FROM target_dictionary
        WHERE chembl_id = ?
        """,
        (TARGET_CHEMBL_ID,),
    ).fetchone()

    if target_row is None:
        raise RuntimeError(f"Target {TARGET_CHEMBL_ID} is absent from the database.")
    if target_row["organism"] != TARGET_ORGANISM:
        raise RuntimeError("Unexpected target organism in the ChEMBL database.")

    return {
        "chembl_version": version_row["name"] if version_row else "unknown",
        "chembl_creation_date": (
            version_row["creation_date"] if version_row else "unknown"
        ),
        "target": dict(target_row),
    }


def load_target_assays(
    connection: sqlite3.Connection,
    target_id: int,
) -> dict[int, dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT assay_id, confidence_score, assay_type
        FROM assays
        WHERE tid = ? AND confidence_score >= 8
        ORDER BY assay_id
        """,
        (target_id,),
    ).fetchall()
    if not rows:
        raise RuntimeError("No high-confidence assays found for the selected target.")
    return {row["assay_id"]: dict(row) for row in rows}


def validate_limit(limit: int | None) -> None:
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be a positive integer.")


def batched(values: list[int], batch_size: int = 400) -> list[list[int]]:
    return [
        values[start : start + batch_size]
        for start in range(0, len(values), batch_size)
    ]


def extract_rows(
    connection: sqlite3.Connection,
    output_path: Path,
    limit: int | None,
    target: dict[str, Any],
    assays: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    validate_limit(limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    molecule_ids: set[int] = set()
    smiles_counts: Counter[str] = Counter()
    assay_counts: Counter[int] = Counter()
    pic50_values: list[float] = []

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(OUTPUT_COLUMNS)

        for assay_batch in batched(list(assays)):
            placeholders = ",".join("?" for _ in assay_batch)
            query = ACTIVITY_QUERY_TEMPLATE.format(
                assay_placeholders=placeholders
            )
            parameters: tuple[Any, ...] = (*assay_batch, ACTIVITY_TYPE)

            for row in connection.execute(query, parameters):
                assay = assays[row["assay_id"]]
                output_row = [row[column] for column in ACTIVITY_COLUMNS]
                output_row.extend(
                    [
                        assay["confidence_score"],
                        assay["assay_type"],
                        target["chembl_id"],
                        target["pref_name"],
                        target["organism"],
                        target["target_type"],
                    ]
                )
                writer.writerow(output_row)
                row_count += 1
                molecule_ids.add(row["molregno"])
                smiles_counts[row["canonical_smiles"]] += 1
                assay_counts[row["assay_id"]] += 1
                pic50_values.append(float(row["pic50"]))

                if limit is not None and row_count >= limit:
                    break
            if limit is not None and row_count >= limit:
                break

    if row_count == 0:
        raise RuntimeError("The strict extraction returned no records.")

    repeated_smiles = sum(count > 1 for count in smiles_counts.values())
    return {
        "records": row_count,
        "unique_molregno": len(molecule_ids),
        "unique_smiles": len(smiles_counts),
        "unique_assays": len(assay_counts),
        "smiles_with_repeated_measurements": repeated_smiles,
        "pic50_min": min(pic50_values),
        "pic50_mean": sum(pic50_values) / len(pic50_values),
        "pic50_max": max(pic50_values),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    connection = connect_read_only(args.db)
    try:
        database_info = validate_database(connection)
        target = database_info["target"]
        assays = load_target_assays(connection, target["tid"])
        statistics = extract_rows(
            connection,
            args.output,
            args.limit,
            target,
            assays,
        )
    finally:
        connection.close()

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "database": str(args.db.resolve()),
        **database_info,
        "filters": {
            "organism": TARGET_ORGANISM,
            "target_chembl_id": TARGET_CHEMBL_ID,
            "target_type": "SINGLE PROTEIN",
            "activity_type": ACTIVITY_TYPE,
            "standard_relation": "=",
            "standard_units": "nM",
            "positive_standard_value": True,
            "pchembl_required": True,
            "minimum_confidence_score": 8,
            "potential_duplicates_excluded": True,
            "invalid_records_excluded": True,
        },
        "limit": args.limit,
        "eligible_high_confidence_assays": len(assays),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "statistics": statistics,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
