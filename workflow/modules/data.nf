// Acquisition and curation stages (ligand-ml image).

process FETCH {
    tag "${name}"
    label 'ml'
    // A persistent snapshot: when <sources_dir>/<name> exists the task is skipped, so cached
    // runs need no network. curate verifies every object's SHA-256 when it reads the snapshot.
    storeDir "${params.sources_dir}"

    input:
    path manifest
    val name

    output:
    path "${name}"

    script:
    """
    ligand-analysis fetch ${manifest} --snapshot-dir '${name}'
    """
}

process CURATE {
    label 'ml'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path manifest
    path snapshot

    output:
    path 'curated'

    script:
    """
    ligand-analysis curate ${manifest} --snapshot-dir ${snapshot} --output-dir curated
    """
}

process LIGAND_INPUTS {
    label 'ml'
    publishDir "${params.outdir}/inputs", mode: 'copy'

    input:
    path curated
    val config_json

    output:
    path 'ligands.tsv'

    script:
    """
    echo '${config_json}' > stage_config.json
    ligand-analysis ligand-inputs ${curated}/candidates.tsv --config stage_config.json --output ligands.tsv
    """
}

process PREDICTION_INPUTS {
    label 'ml'
    publishDir "${params.outdir}/inputs", mode: 'copy'

    input:
    path curated
    val config_json

    output:
    path 'prediction_ligands.tsv'

    script:
    """
    echo '${config_json}' > stage_config.json
    ligand-analysis prediction-inputs ${curated}/ligands.tsv --candidates ${curated}/candidates.tsv \\
        --config stage_config.json --output prediction_ligands.tsv
    """
}
