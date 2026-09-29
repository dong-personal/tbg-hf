import argparse
import csv
import re
from datetime import datetime
from pathlib import Path


DEFAULT_DB = Path("data/band_index.csv")
DEFAULT_PATTERN = "data/band*.txt"
TIMESTAMP_RE = re.compile(r"band(?P<timestamp>\d{14})\.txt$")
META_RE = re.compile(r"^%\s*(?P<section>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<body>.+?)\s*$")
PAIR_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>[-+0-9.eE]+)")
BASE_COLUMNS = ["timestamp", "datetime", "file", "label"]


def parse_timestamp(path):
    match = TIMESTAMP_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot find timestamp in filename: {path}")

    timestamp = match.group("timestamp")
    dt = datetime.strptime(timestamp, "%Y%m%d%H%M%S").isoformat(sep=" ")
    return timestamp, dt


def parse_metadata(path):
    metadata = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = META_RE.match(line)
            if not match:
                continue

            section = match.group("section")
            pairs = {
                pair.group("key"): pair.group("value")
                for pair in PAIR_RE.finditer(match.group("body"))
            }
            if pairs:
                metadata[section] = pairs

    if "given" not in metadata:
        raise ValueError(f"No '% given:' line found in file: {path}")

    return metadata


def make_row(path, label=""):
    path = Path(path)
    timestamp, dt = parse_timestamp(path)
    metadata = parse_metadata(path)
    row = {
        "timestamp": timestamp,
        "datetime": dt,
        "file": str(path),
        "label": label,
    }

    for section, values in metadata.items():
        for key, value in values.items():
            row[f"{section}_{key}"] = value

    return row


def numeric_equal(left, right, tolerance=1e-12):
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return str(left) == str(right)


def normalize_params(params, prefix="given"):
    normalized = {}
    for key, value in params.items():
        column = key if key.startswith(f"{prefix}_") else f"{prefix}_{key}"
        normalized[column] = str(value)
    return normalized


def row_matches_params(row, params, prefix="given", tolerance=1e-12):
    normalized = normalize_params(params, prefix=prefix)
    for column, expected in normalized.items():
        if column not in row:
            return False
        if not numeric_equal(row[column], expected, tolerance=tolerance):
            return False
    return True


def collect_fieldnames(rows):
    extra_columns = sorted(
        {
            column
            for row in rows
            for column in row
            if column not in BASE_COLUMNS
        }
    )
    return [*BASE_COLUMNS, *extra_columns]


def read_db(db_path=DEFAULT_DB):
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    with db_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_db(rows, db_path=DEFAULT_DB):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: row["timestamp"])
    fieldnames = collect_fieldnames(rows)

    with db_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_db(files=None, db_path=DEFAULT_DB):
    if files is None:
        files = sorted(Path().glob(DEFAULT_PATTERN))
    rows = [make_row(path) for path in sorted(files)]
    write_db(rows, db_path=db_path)
    return rows


def get_params_by_timestamp(timestamp, db_path=DEFAULT_DB):
    timestamp = str(timestamp)
    for row in read_db(db_path):
        if row["timestamp"] == timestamp:
            return dict(row)
    return None


def find_timestamps_by_params(params, db_path=DEFAULT_DB, prefix="given", tolerance=1e-12):
    rows = read_db(db_path)
    return [
        row["timestamp"]
        for row in rows
        if row_matches_params(row, params, prefix=prefix, tolerance=tolerance)
    ]


def find_rows_by_params(params, db_path=DEFAULT_DB, prefix="given", tolerance=1e-12):
    rows = read_db(db_path)
    return [
        row
        for row in rows
        if row_matches_params(row, params, prefix=prefix, tolerance=tolerance)
    ]


def upsert_band_file(
    path,
    db_path=DEFAULT_DB,
    on_exists="ignore",
    match_params=None,
    match_prefix="given",
    tolerance=1e-12,
    label=None,
):
    if on_exists not in {"ignore", "overwrite"}:
        raise ValueError("on_exists must be 'ignore' or 'overwrite'")

    new_row = make_row(path, label=label or "")
    rows = read_db(db_path)
    params = match_params or {
        key.removeprefix(f"{match_prefix}_"): value
        for key, value in new_row.items()
        if key.startswith(f"{match_prefix}_")
    }

    matched_index = None
    for index, row in enumerate(rows):
        if row_matches_params(row, params, prefix=match_prefix, tolerance=tolerance):
            matched_index = index
            break

    if matched_index is None:
        rows.append(new_row)
        status = "inserted"
    elif on_exists == "ignore":
        return "ignored"
    else:
        if label is None:
            new_row["label"] = rows[matched_index].get("label", "")
        rows[matched_index] = new_row
        status = "overwritten"

    write_db(rows, db_path=db_path)
    return status


def update_label_by_timestamp(timestamp, label, db_path=DEFAULT_DB):
    rows = read_db(db_path)
    for row in rows:
        if row["timestamp"] == str(timestamp):
            row["label"] = str(label)
            write_db(rows, db_path=db_path)
            return True
    return False


def update_label_by_params(params, label, db_path=DEFAULT_DB, prefix="given", tolerance=1e-12):
    rows = read_db(db_path)
    updated = 0
    for row in rows:
        if row_matches_params(row, params, prefix=prefix, tolerance=tolerance):
            row["label"] = str(label)
            updated += 1

    if updated:
        write_db(rows, db_path=db_path)
    return updated


def replace_label(old_label, new_label, db_path=DEFAULT_DB):
    rows = read_db(db_path)
    updated = 0
    for row in rows:
        if row.get("label", "") == str(old_label):
            row["label"] = str(new_label)
            updated += 1

    if updated:
        write_db(rows, db_path=db_path)
    return updated


def parse_key_values(items):
    params = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected key=value, got: {item}")
        key, value = item.split("=", 1)
        params[key.strip()] = value.strip()
    return params


def main():
    parser = argparse.ArgumentParser(
        description="Build a timestamp-to-parameter database from band*.txt files."
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help=f"Input band*.txt files. Defaults to {DEFAULT_PATTERN}.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_DB,
        help=f"Output database CSV. Defaults to {DEFAULT_DB}.",
    )
    args = parser.parse_args()

    files = args.files or sorted(Path().glob(DEFAULT_PATTERN))
    if not files:
        raise SystemExit("No band*.txt files found.")

    rows = build_db(files=files, db_path=args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
