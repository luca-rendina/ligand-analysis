"""Tab-separated tables validated against the row schemas in ligand_analysis/schemas.

Column order follows the schema. Empty cells mean null; booleans are true/false.
"""

import csv
import io
import re

import jsonschema

from .config import ConfigError, load_schema
from .fileio import write_text_atomic


def _types(prop):
    kind = prop.get("type", "string")
    return [kind] if isinstance(kind, str) else list(kind)


def _format(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def _parse(text, prop):
    types = _types(prop)
    if text == "" and "null" in types:
        return None
    if "integer" in types and re.fullmatch(r"-?\d+", text):
        return int(text)
    if "number" in types and re.fullmatch(r"-?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", text):
        return float(text)
    if "boolean" in types and text in ("true", "false"):
        return text == "true"
    return text


def _check(rows, schema, what):
    validator = jsonschema.Draft202012Validator(schema)
    errors = [f"row {number}: {'/'.join(map(str, error.absolute_path)) or '<row>'}: {error.message}"
              for number, row in enumerate(rows, start=2)
              for error in validator.iter_errors(row)]
    if errors:
        shown = "\n  ".join(errors[:20])
        raise ConfigError(f"{what} has {len(errors)} schema error(s) (rows count from the header line 1):\n  {shown}")


def write_table(path, rows, schema_name):
    schema = load_schema(schema_name)
    rows = list(rows)
    _check(rows, schema, f"{schema_name} table for {path}")
    columns = list(schema["properties"])
    stream = io.StringIO()
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(columns)
    writer.writerows([_format(row.get(column)) for column in columns] for row in rows)
    write_text_atomic(path, stream.getvalue())


def read_table(path, schema_name):
    schema = load_schema(schema_name)
    properties = schema["properties"]
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        header = reader.fieldnames or []
        unknown = sorted(set(header) - properties.keys())
        missing = sorted(set(schema.get("required", [])) - set(header))
        if unknown or missing:
            raise ConfigError(f"{path} columns do not match {schema_name}: "
                              f"missing {missing or 'none'}, unexpected {unknown or 'none'}")
        rows = []
        for row in reader:
            if None in row or None in row.values():
                raise ConfigError(f"{path} line {reader.line_num}: wrong number of fields")
            rows.append({name: _parse(value, properties[name]) for name, value in row.items()})
    _check(rows, schema, str(path))
    return rows
