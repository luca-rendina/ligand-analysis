"""Local directory configuration, YAML manifests and JSON Schema validation."""

from functools import cache
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path

import jsonschema
import yaml

DATA_DIR_ENV = "LIGAND_ANALYSIS_DATA_DIR"


class ConfigError(ValueError):
    """Invalid configuration, manifest or table content."""


def data_dir(explicit=None):
    """Return the local data directory from --data-dir or LIGAND_ANALYSIS_DATA_DIR."""
    value = explicit or os.environ.get(DATA_DIR_ENV)
    if not value:
        raise ConfigError(f"Set {DATA_DIR_ENV} or pass --data-dir.")
    return Path(value)


@cache
def load_schema(name):
    resource = files("ligand_analysis") / "schemas" / f"{name}.schema.json"
    return json.loads(resource.read_text(encoding="utf-8"))


def schema_errors(instance, schema):
    validator = jsonschema.Draft202012Validator(schema)
    return sorted(f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
                  for error in validator.iter_errors(instance))


def validate(instance, schema_name, what):
    errors = schema_errors(instance, load_schema(schema_name))
    if errors:
        raise ConfigError(f"{what} is invalid:\n  " + "\n  ".join(errors))


def load_manifest(path):
    """Load and validate a versioned source manifest."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        try:
            manifest = yaml.safe_load(handle)
        except yaml.YAMLError as error:
            raise ConfigError(f"{path} is not valid YAML: {error}") from error
    validate(manifest, "source_manifest", str(path))
    return manifest


def manifest_sha256(path):
    """SHA-256 of a manifest with LF line endings, so CRLF (Windows) checkouts give the same digest."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
