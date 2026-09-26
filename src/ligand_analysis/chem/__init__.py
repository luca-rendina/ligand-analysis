"""Chemistry stages that need the ligand-chem image (RDKit, molscrub, Meeko, Vina, PDBFixer, ODDT).

Modules import their toolkits lazily so the package still imports in the ligand-ml image.
"""


class ChemistryError(RuntimeError):
    """A chemistry stage cannot produce a valid output for its declared inputs."""


def safe_key(identifier):
    """File-system-safe name for a ligand or state identifier."""
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in identifier)


def distribution_version(name):
    """Installed version of a distribution, or None."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None
