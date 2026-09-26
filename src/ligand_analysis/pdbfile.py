"""Minimal fixed-column parsing of legacy PDB files (atoms, DBREF, SEQADV, header facts)."""

from dataclasses import dataclass
import re

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
WATER_NAMES = {"HOH", "WAT", "DOD"}


class PdbError(ValueError):
    """The PDB text lacks records the preparation needs."""


@dataclass(frozen=True)
class Atom:
    record: str
    serial: int
    name: str
    altloc: str
    resname: str
    chain: str
    resnum: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: float
    element: str
    line: str

    @property
    def residue_key(self):
        return self.chain, self.resnum, self.icode

    @property
    def is_hydrogen(self):
        return self.element in ("H", "D")


@dataclass(frozen=True)
class DbRef:
    chain: str
    seq_begin: int
    ins_begin: str
    seq_end: int
    ins_end: str
    database: str
    accession: str
    db_begin: int
    db_end: int

    def db_position(self, resnum, icode=""):
        """Database position of an author residue inside this segment, or None."""
        if icode or not self.seq_begin <= resnum <= self.seq_end:
            return None
        return self.db_begin + resnum - self.seq_begin


@dataclass(frozen=True)
class SeqAdv:
    resname: str
    chain: str
    resnum: int | None
    icode: str
    database: str
    accession: str
    db_resname: str
    db_resnum: int | None
    comment: str


def _int(text):
    text = text.strip()
    return int(text) if re.fullmatch(r"-?\d+", text) else None


def _element(line, name):
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element:
        return element
    letters = re.sub(r"[^A-Za-z]", "", name)
    return letters[:1].upper() if letters else ""


def parse_atoms(text):
    """ATOM and HETATM records of the first model."""
    atoms = []
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        name = line[12:16].strip()
        atoms.append(Atom(
            record=line[:6].strip(), serial=_int(line[6:11]) or 0, name=name, altloc=line[16].strip(),
            resname=line[17:20].strip(), chain=line[21].strip(), resnum=int(line[22:26]), icode=line[26].strip(),
            x=float(line[30:38]), y=float(line[38:46]), z=float(line[46:54]),
            occupancy=float(line[54:60]) if line[54:60].strip() else 1.0, element=_element(line, name), line=line))
    if not atoms:
        raise PdbError("no ATOM or HETATM records")
    return atoms


def parse_dbref(text):
    refs = []
    for line in text.splitlines():
        if line.startswith("DBREF "):
            line = line.ljust(68)
            refs.append(DbRef(chain=line[12].strip(), seq_begin=int(line[14:18]), ins_begin=line[18].strip(),
                              seq_end=int(line[20:24]), ins_end=line[24].strip(), database=line[26:32].strip(),
                              accession=line[33:41].strip(), db_begin=int(line[55:60]), db_end=int(line[62:67])))
    return refs


def parse_seqadv(text):
    records = []
    for line in text.splitlines():
        if line.startswith("SEQADV"):
            line = line.ljust(70)
            records.append(SeqAdv(resname=line[12:15].strip(), chain=line[16].strip(), resnum=_int(line[18:22]),
                                  icode=line[22].strip(), database=line[24:28].strip(), accession=line[29:38].strip(),
                                  db_resname=line[39:42].strip(), db_resnum=_int(line[43:48]),
                                  comment=line[49:70].strip()))
    return records


def parse_header(text):
    """Experimental method and resolution (Angstrom) when present."""
    method, resolution = None, None
    for line in text.splitlines():
        if line.startswith("EXPDTA"):
            method = line[10:].strip()
        match = re.match(r"REMARK   2 RESOLUTION\.\s+([0-9.]+)\s+ANGSTROMS", line)
        if match:
            resolution = float(match.group(1))
    return {"method": method, "resolution_angstrom": resolution}
