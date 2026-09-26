"""Fetch pinned sources and curate receptor-specific agonist/antagonist candidates.

Labels come only from explicit (GtoPdb Type, Action) rules in the manifest, joined through
GtoPdb ligand IDs and the receptor UniProt accession. Filenames, docking scores, conformers
and poses are never used. Terms without a rule stay unmapped; ligands whose annotations mix
mapped and unmapped terms (ambiguous) or both labels (conflicting) are excluded.
"""

import collections
import csv
from dataclasses import dataclass
import hashlib
import html
import io
import json
from pathlib import Path
import re

from . import __version__
from .config import ConfigError
from .fileio import write_json
from .labels import CLASS_NAMES
from .tables import write_table

PUBCHEM_BATCH = 200
INCHIKEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
RELEASE = re.compile(r"GtoPdb Version: (\S+) - published: (\S+)")
OUTPUT_FILES = ("receptor.tsv", "ligands.tsv", "annotations.tsv", "issues.tsv", "exclusions.tsv",
                "candidates.tsv", "curation_report.json")
SELECTION_ORDER = ("approved drugs first, then ascending GtoPdb ligand ID; at most one ligand "
                   "per InChIKey connectivity block across both classes")


class CurationError(RuntimeError):
    """Source records contradict the manifest or cannot be parsed."""


@dataclass
class SourceData:
    release: str
    published: str
    interactions: list
    ligand_records: dict
    uniprot: dict
    pubchem: dict
    source_urls: list


def _gtopdb_table(data, what):
    first, _, rest = data.decode("utf-8-sig").partition("\n")
    match = RELEASE.search(first)
    if not match:
        raise CurationError(f"{what}: missing '# GtoPdb Version' header line")
    return match.group(1), match.group(2), list(csv.DictReader(io.StringIO(rest)))


def _clean(text):
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def _flag(value):
    # ligands.csv uses "yes"/"", interactions.csv uses "true"/"false".
    return value.strip().lower() in ("yes", "true")


def _optional(value):
    value = (value or "").strip()
    return value or None


def _json(data, what):
    try:
        return json.loads(data)
    except ValueError as error:
        raise CurationError(f"{what} is not valid JSON: {error}") from error


def load_sources(manifest, snapshot):
    """Read every manifest source through the snapshot (downloading only if allowed)."""
    sources = manifest["sources"]
    releases, tables, urls = {}, {}, []
    for key in ("gtopdb_interactions", "gtopdb_ligands"):
        source = sources[key]
        release, published, rows = _gtopdb_table(snapshot.get(key, source["url"], source["sha256"]), key)
        if release != source["release"]:
            raise CurationError(f"{key}: file is GtoPdb {release}, manifest pins {source['release']}")
        releases[key], tables[key] = (release, published), rows
        urls.append(source["url"])
    if releases["gtopdb_interactions"] != releases["gtopdb_ligands"]:
        raise CurationError(f"GtoPdb files come from different releases: {releases}")

    uniprot_source = sources["uniprot_entry"]
    uniprot = _json(snapshot.get("uniprot_entry", uniprot_source["url"], uniprot_source.get("sha256")),
                    uniprot_source["url"])
    urls.append(uniprot_source["url"])

    accession = manifest["receptor"]["uniprot_accession"]
    interactions = [row for row in tables["gtopdb_interactions"] if row["Target UniProt ID"].strip() == accession]
    records = collections.defaultdict(list)
    for row in tables["gtopdb_ligands"]:
        records[row["Ligand ID"].strip()].append(row)
    annotated = {row["Ligand ID"].strip() for row in interactions}
    cids = sorted({int(row["PubChem CID"]) for ligand in annotated for row in records.get(ligand, [])
                   if row["PubChem CID"].strip().isdigit()})

    pubchem = {}
    for start in range(0, len(cids), PUBCHEM_BATCH):
        url = sources["pubchem_identity"]["url"].replace(
            "{cids}", ",".join(map(str, cids[start:start + PUBCHEM_BATCH])))
        payload = _json(snapshot.get("pubchem_identity", url), url)
        if "PropertyTable" not in payload:
            raise CurationError(f"Unexpected PubChem response for {url}: {str(payload)[:200]}")
        for item in payload["PropertyTable"]["Properties"]:
            pubchem[int(item["CID"])] = item.get("InChIKey")
        urls.append(url)
    release, published = releases["gtopdb_interactions"]
    return SourceData(release, published, interactions, dict(records), uniprot, pubchem, urls)


def fetch(manifest, snapshot):
    """Populate the snapshot with every source curation needs, then save its index."""
    load_sources(manifest, snapshot)
    snapshot.save()
    return snapshot.events


def _receptor(manifest, data):
    spec = manifest["receptor"]
    accession = spec["uniprot_accession"]
    problems = []
    if not data.interactions:
        problems.append(f"GtoPdb has no interactions with Target UniProt ID {accession}")
    targets = {(row["Target ID"], row["Target Gene Symbol"], row["Target Species"], row["Target"])
               for row in data.interactions}
    for target_id, gene, species, _ in sorted(targets):
        if (target_id, gene, species) != (str(spec["gtopdb_target_id"]), spec["gene_symbol"], spec["species"]):
            problems.append(f"GtoPdb target {target_id}/{gene}/{species} differs from the manifest "
                            f"{spec['gtopdb_target_id']}/{spec['gene_symbol']}/{spec['species']}")
    uniprot = data.uniprot
    genes = [gene.get("geneName", {}).get("value") for gene in uniprot.get("genes", [])]
    if uniprot.get("primaryAccession") != accession:
        problems.append(f"UniProt entry is {uniprot.get('primaryAccession')}, expected {accession}")
    if uniprot.get("organism", {}).get("taxonId") != spec["taxon_id"]:
        problems.append(f"UniProt taxon {uniprot.get('organism', {}).get('taxonId')}, expected {spec['taxon_id']}")
    if spec["gene_symbol"] not in genes:
        problems.append(f"UniProt genes {genes} do not include {spec['gene_symbol']}")
    if problems:
        raise CurationError("Receptor identity check failed:\n  " + "\n  ".join(problems))
    return {
        "receptor_id": accession,
        "uniprot_accession": accession,
        "uniprot_entry_name": uniprot["uniProtkbId"],
        "gene_symbol": spec["gene_symbol"],
        "name": _clean(min(targets)[3]),
        "species": spec["species"],
        "taxon_id": spec["taxon_id"],
        "gtopdb_target_id": spec["gtopdb_target_id"],
        "sequence_length": uniprot["sequence"]["length"],
        "sequence_md5": uniprot["sequence"]["md5"],
    }


def _label_rules(manifest):
    rules = {}
    for rule in manifest["label_mapping"]:
        key = (rule["type"], rule["action"])
        if key in rules and rules[key] != rule["label"]:
            raise ConfigError(f"label_mapping maps {key} to both labels")
        rules[key] = rule["label"]
    return rules


def _issue(severity, category, issue_type, subject, related, details):
    return {"severity": severity, "category": category, "issue_type": issue_type, "subject_id": subject,
            "related_ids": "|".join(related) if related else None, "details": details}


def _term(annotation):
    return f"{annotation['original_type']} / {annotation['original_action']}"


def _annotations(data, receptor_id, rules, issues):
    annotations = {}
    for row in data.interactions:
        raw = "\x1f".join(f"{key}={value}" for key, value in row.items())
        annotation_id = "gtopdb:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        ligand_id = f"iuphar.ligand:{row['Ligand ID'].strip()}"
        if annotation_id in annotations:
            issues.append(_issue("warning", "annotation", "duplicate_annotation", annotation_id, [ligand_id],
                                 "identical GtoPdb interaction rows; counted once"))
            continue
        label = rules.get((row["Type"], row["Action"]))
        annotations[annotation_id] = {
            "annotation_id": annotation_id,
            "ligand_id": ligand_id,
            "receptor_id": receptor_id,
            "source_release": f"GtoPdb {data.release}",
            "target_species": row["Target Species"],
            "original_type": row["Type"],
            "original_action": row["Action"],
            "action_comment": _optional(row["Action comment"]),
            "label": label,
            "label_status": "unmapped" if label is None else "mapped",
            "pair_status": None,
            "selectivity": _optional(row["Selectivity"]),
            "endogenous": _flag(row["Endogenous"]),
            "primary_target": _flag(row["Primary Target"]),
            "affinity_units": _optional(row["Affinity Units"]),
            "affinity_low": _optional(row["Affinity Low"]),
            "affinity_median": _optional(row["Affinity Median"]),
            "affinity_high": _optional(row["Affinity High"]),
            "assay_description": _optional(row["Assay Description"]),
            "receptor_site": _optional(row["Receptor Site"]),
            "ligand_context": _optional(row["Ligand Context"]),
            "pubmed_ids": _optional(row["PubMed ID"]),
        }
    by_ligand = collections.defaultdict(list)
    for annotation in annotations.values():
        by_ligand[annotation["ligand_id"]].append(annotation)
    for ligand_id, items in by_ligand.items():
        labels = {item["label"] for item in items if item["label"] is not None}
        unmapped = any(item["label"] is None for item in items)
        status = ("conflicting" if len(labels) > 1 else "ambiguous" if labels and unmapped
                  else "consistent" if labels else "unmapped")
        for item in items:
            item["pair_status"] = status
        terms = "; ".join(sorted({_term(item) for item in items}))
        related = sorted(item["annotation_id"] for item in items)
        if status == "conflicting":
            issues.append(_issue("error", "annotation", "conflicting_labels", ligand_id, related,
                                 f"annotations map to both labels: {terms}"))
        elif status == "ambiguous":
            issues.append(_issue("warning", "annotation", "ambiguous_labels", ligand_id, related,
                                 f"mapped and unmapped terms: {terms}"))
    ordered = sorted(annotations.values(), key=lambda item: (_numeric_id(item["ligand_id"]), item["annotation_id"]))
    return ordered, by_ligand


def _numeric_id(ligand_id):
    return int(ligand_id.rsplit(":", 1)[1])


def _ligands(data, ligand_ids, issues):
    ligands, problems = [], collections.defaultdict(list)
    for ligand_id in sorted(ligand_ids, key=_numeric_id):
        records = data.ligand_records.get(str(_numeric_id(ligand_id)), [])
        if not records:
            issues.append(_issue("error", "ligand", "missing_ligand_record", ligand_id, None,
                                 "annotated ligand has no row in ligands.csv"))
            problems[ligand_id].append(("missing_ligand_record", "no row in ligands.csv"))
            continue
        if len(records) > 1:
            issues.append(_issue("error", "ligand", "duplicate_ligand_record", ligand_id, None,
                                 f"{len(records)} rows in ligands.csv; first row reported"))
            problems[ligand_id].append(("duplicate_ligand_record", f"{len(records)} rows in ligands.csv"))
        record = records[0]
        smiles, inchikey = _optional(record["SMILES"]), _optional(record["InChIKey"])
        cid = int(record["PubChem CID"]) if record["PubChem CID"].strip().isdigit() else None
        pubchem_key = data.pubchem.get(cid) if cid else None
        if not smiles or not inchikey:
            status, details = "missing_structure", "GtoPdb has no SMILES or InChIKey"
            issues.append(_issue("error", "ligand", "missing_structure", ligand_id, None, details))
            problems[ligand_id].append(("missing_structure", details))
        elif not INCHIKEY.match(inchikey):
            status, details = "invalid_inchikey", f"malformed InChIKey {inchikey!r}"
            issues.append(_issue("error", "ligand", "invalid_inchikey", ligand_id, None, details))
            problems[ligand_id].append(("invalid_inchikey", details))
        elif pubchem_key is None:
            status, details = "unverified", f"no PubChem InChIKey for CID {cid}"
            issues.append(_issue("warning", "ligand", "pubchem_unverified", ligand_id, None, details))
            problems[ligand_id].append(("identity_unverified", details))
        elif pubchem_key != inchikey:
            status, details = "mismatch", f"GtoPdb {inchikey} but PubChem CID {cid} is {pubchem_key}"
            issues.append(_issue("error", "ligand", "pubchem_mismatch", ligand_id, None, details))
            problems[ligand_id].append(("identity_mismatch", details))
        else:
            status = "verified"
        valid_key = status in ("verified", "unverified", "mismatch")
        ligands.append({
            "ligand_id": ligand_id,
            "gtopdb_ligand_id": _numeric_id(ligand_id),
            "name": _clean(record["Name"]),
            "ligand_type": record["Type"],
            "approved": _flag(record["Approved"]),
            "withdrawn": _flag(record["Withdrawn"]),
            "labelled": _flag(record["Labelled"]),
            "radioactive": _flag(record["Radioactive"]),
            "smiles": smiles,
            "inchikey": inchikey,
            "connectivity_key": inchikey[:14] if valid_key else None,
            "pubchem_cid": cid,
            "pubchem_inchikey": pubchem_key,
            "chembl_id": _optional(record["ChEMBL ID"]),
            "identity_status": status,
        })

    by_key, by_block = collections.defaultdict(list), collections.defaultdict(list)
    for ligand in ligands:
        if ligand["connectivity_key"]:
            by_key[ligand["inchikey"]].append(ligand["ligand_id"])
            by_block[ligand["connectivity_key"]].append(ligand)
    for inchikey, ids in sorted(by_key.items()):
        if len(ids) > 1:
            issues.append(_issue("warning", "ligand", "duplicate_structure", ids[0], ids[1:],
                                 f"GtoPdb ligands share InChIKey {inchikey}"))
    for block, members in sorted(by_block.items()):
        if len({member["inchikey"] for member in members}) > 1:
            ids = [member["ligand_id"] for member in members]
            issues.append(_issue("info", "ligand", "shared_connectivity", ids[0], ids[1:],
                                 f"InChIKey connectivity block {block} shared by stereoisomers, isotopologues "
                                 "or other forms; treated as one group for selection"))
    return ligands, problems


def curate(manifest, snapshot, output_dir, manifest_sha256=None):
    """Write receptor, ligand, annotation, issue, exclusion and candidate tables."""
    data = load_sources(manifest, snapshot)
    receptor = _receptor(manifest, data)
    receptor_id = receptor["receptor_id"]
    rules = _label_rules(manifest)
    inclusion = manifest["inclusion"]
    issues = []
    annotations, by_ligand = _annotations(data, receptor_id, rules, issues)
    ligands, problems = _ligands(data, by_ligand.keys(), issues)
    ligand_by_id = {ligand["ligand_id"]: ligand for ligand in ligands}

    reasons = collections.defaultdict(list)
    for ligand_id, items in by_ligand.items():
        status = items[0]["pair_status"]
        terms = "; ".join(sorted({_term(item) for item in items}))
        if status != "consistent":
            reasons[ligand_id].append(({"unmapped": "unmapped_action", "ambiguous": "ambiguous_label",
                                        "conflicting": "conflicting_label"}[status], terms))
        reasons[ligand_id].extend(problems.get(ligand_id, []))
        ligand = ligand_by_id.get(ligand_id)
        if ligand is None:
            continue
        if ligand["ligand_type"] not in inclusion["ligand_types"]:
            reasons[ligand_id].append(("ligand_type", ligand["ligand_type"]))
        if inclusion["exclude_labelled"] and ligand["labelled"]:
            reasons[ligand_id].append(("labelled", "GtoPdb marks the ligand as labelled"))
        if inclusion["exclude_radioactive"] and ligand["radioactive"]:
            reasons[ligand_id].append(("radioactive", "GtoPdb marks the ligand as radioactive"))
        if not inclusion["require_pubchem_match"]:
            reasons[ligand_id] = [reason for reason in reasons[ligand_id] if reason[0] != "identity_unverified"]

    label_of = {ligand_id: items[0]["label"] for ligand_id, items in by_ligand.items()}
    groups = collections.defaultdict(list)
    for ligand_id, ligand in ligand_by_id.items():
        if not reasons[ligand_id]:
            groups[ligand["connectivity_key"]].append(ligand_id)
    for block, ids in sorted(groups.items()):
        if len({label_of[ligand_id] for ligand_id in ids}) > 1:
            ids = sorted(ids, key=_numeric_id)
            issues.append(_issue("warning", "annotation", "connectivity_label_conflict", ids[0], ids[1:],
                                 f"ligands with connectivity block {block} carry both labels"))
            for ligand_id in ids:
                reasons[ligand_id].append(("connectivity_label_conflict", f"connectivity block {block}"))

    def name_of(ligand_id):
        ligand = ligand_by_id.get(ligand_id)
        return ligand["name"] if ligand else ""

    exclusions = [{"ligand_id": ligand_id, "receptor_id": receptor_id, "name": name_of(ligand_id),
                   "reason": reason, "details": details}
                  for ligand_id in sorted(reasons, key=_numeric_id) for reason, details in reasons[ligand_id]]

    eligible = sorted((ligand for ligand_id, ligand in ligand_by_id.items() if not reasons[ligand_id]),
                      key=lambda ligand: (not ligand["approved"], ligand["gtopdb_ligand_id"]))
    max_per_class = manifest["selection"]["max_per_class"]
    chosen_blocks, counts, candidates = {}, collections.Counter(), []
    for ligand in eligible:
        ligand_id, label = ligand["ligand_id"], label_of[ligand["ligand_id"]]
        items = by_ligand[ligand_id]
        block = ligand["connectivity_key"]
        note, rank = None, None
        if block in chosen_blocks:
            note = f"same connectivity block as selected {chosen_blocks[block]}"
        elif counts[label] >= max_per_class:
            note = f"class limit of {max_per_class} reached"
        else:
            counts[label] += 1
            rank = counts[label]
            chosen_blocks[block] = ligand_id
        pubmed = sorted({pmid for item in items if item["pubmed_ids"] for pmid in item["pubmed_ids"].split("|")},
                        key=int)
        candidates.append({
            "ligand_id": ligand_id, "receptor_id": receptor_id, "label": label,
            "class_name": CLASS_NAMES[label], "selected": rank is not None, "selection_rank": rank,
            "selection_note": note, "name": ligand["name"], "approved": ligand["approved"],
            "smiles": ligand["smiles"], "inchikey": ligand["inchikey"], "connectivity_key": block,
            "pubchem_cid": ligand["pubchem_cid"], "chembl_id": ligand["chembl_id"],
            "original_terms": "; ".join(sorted({_term(item) for item in items})),
            "pubmed_ids": "|".join(pubmed) or None, "n_annotations": len(items),
        })
    candidates.sort(key=lambda row: (row["label"], not row["selected"], row["selection_rank"] or 0,
                                     not row["approved"], _numeric_id(row["ligand_id"])))
    issue_order = {"error": 0, "warning": 1, "info": 2}
    issues.sort(key=lambda row: (issue_order[row["severity"]], row["issue_type"], row["subject_id"]))

    output_dir = Path(output_dir)
    write_table(output_dir / "receptor.tsv", [receptor], "receptor")
    write_table(output_dir / "ligands.tsv", ligands, "ligand")
    write_table(output_dir / "annotations.tsv", annotations, "annotation")
    write_table(output_dir / "issues.tsv", issues, "issue")
    write_table(output_dir / "exclusions.tsv", exclusions, "exclusion")
    write_table(output_dir / "candidates.tsv", candidates, "candidate")

    def by_class(rows):
        counter = collections.Counter(row["class_name"] for row in rows)
        return {name: counter.get(name, 0) for name in CLASS_NAMES.values()}

    report = {
        "report_type": "curation",
        "package_version": __version__,
        "manifest": {"name": manifest["name"], "sha256": manifest_sha256},
        "receptor": receptor,
        "gtopdb_release": {"version": data.release, "published": data.published},
        "sources": [{key: snapshot.entry(url)[key] for key in ("source_id", "url", "sha256", "bytes", "retrieved_utc")}
                    for url in data.source_urls],
        "rules": {
            "label_mapping": manifest["label_mapping"],
            "unmapped_terms": "excluded; a ligand with mapped and unmapped terms is ambiguous and excluded",
            "join_keys": ["GtoPdb ligand ID", "Target UniProt ID"],
            "inclusion": inclusion,
            "selection": {"max_per_class": max_per_class, "order": SELECTION_ORDER},
        },
        "class_mapping": {str(label): name for label, name in CLASS_NAMES.items()},
        "counts": {
            "annotations": len(annotations),
            "annotations_by_term": dict(sorted(collections.Counter(_term(item) for item in annotations).items())),
            "annotated_ligands": len(by_ligand),
            "ligands_by_pair_status": dict(sorted(collections.Counter(
                items[0]["pair_status"] for items in by_ligand.values()).items())),
            "excluded_ligands": sum(1 for ligand_id in by_ligand if reasons[ligand_id]),
            "exclusions_by_reason": dict(sorted(collections.Counter(row["reason"] for row in exclusions).items())),
            "issues_by_type": dict(sorted(collections.Counter(
                f"{row['severity']}:{row['issue_type']}" for row in issues).items())),
            "eligible_by_class": by_class(candidates),
            "selected_by_class": by_class([row for row in candidates if row["selected"]]),
        },
        "outputs": list(OUTPUT_FILES),
    }
    write_json(output_dir / "curation_report.json", report)
    return report
