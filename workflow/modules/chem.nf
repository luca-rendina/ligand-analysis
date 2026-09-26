// Chemistry stages (ligand-chem image). None of them reads a pharmacology label.

process PREPARE_RECEPTOR {
    label 'chem'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path manifest
    path snapshot
    val config_json

    output:
    path 'receptor'

    script:
    """
    echo '${config_json}' > stage_config.json
    ligand-analysis prepare-receptor ${manifest} --snapshot-dir ${snapshot} --config stage_config.json \\
        --output-dir receptor
    """
}

process REDOCK_REFERENCE {
    label 'chem'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path receptor
    val config_json

    output:
    path 'redocking'

    script:
    """
    echo '${config_json}' > stage_config.json
    ligand-analysis redock-reference --receptor-dir ${receptor} --config stage_config.json --cpu ${task.cpus} \\
        --output-dir redocking
    """
}

// Outputs of the per-table stages sit under <prefix>/ ('labelled' or 'prediction'), so one
// static publishDir keeps the two ligand sets apart.

process PREPARE_LIGANDS {
    tag "${prefix}"
    label 'chem'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    val prefix
    path ligands
    val config_json

    output:
    path "${prefix}/prepared"

    script:
    """
    echo '${config_json}' > stage_config.json
    ligand-analysis prepare-ligands ${ligands} --config stage_config.json --output-dir ${prefix}/prepared
    """
}

process DOCK {
    tag "${prefix}:${ligand_id}"
    label 'chem'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    tuple val(prefix), val(ligand_id), val(key)
    path receptor
    path prepared
    val config_json

    output:
    path "${prefix}/docking/docking_${key}"

    script:
    """
    echo '${config_json}' > stage_config.json
    ligand-analysis dock --receptor-dir ${receptor} --ligands-dir ${prepared} --ligand-id '${ligand_id}' \\
        --cpu ${task.cpus} --config stage_config.json --output-dir ${prefix}/docking/docking_${key}
    """
}

process SELECT_POSES {
    tag "${prefix}"
    label 'chem'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    val prefix
    path ligands
    path prepared
    path docking, stageAs: 'docking/*'
    path receptor
    val config_json

    output:
    path "${prefix}/poses"

    script:
    """
    echo '${config_json}' > stage_config.json
    ligand-analysis select-poses ${ligands} --ligands-dir ${prepared} --docking-dir ${docking} \\
        --receptor-dir ${receptor} --config stage_config.json --output-dir ${prefix}/poses
    """
}

process FEATURIZE_PLEC {
    tag "${prefix}"
    label 'chem'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    val prefix
    path poses
    path receptor
    val config_json

    output:
    tuple val('plec'), path("${prefix}/features/plec")

    script:
    """
    echo '${config_json}' > stage_config.json
    ligand-analysis featurize plec --poses-dir ${poses} --receptor-dir ${receptor} --config stage_config.json \\
        --output-dir ${prefix}/features/plec
    """
}

process FEATURIZE_MORGAN {
    tag "${prefix}"
    label 'chem'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    val prefix
    path prepared
    path curated
    val config_json

    output:
    tuple val('morgan'), path("${prefix}/features/morgan")

    script:
    """
    echo '${config_json}' > stage_config.json
    ligand-analysis featurize morgan --ligands-dir ${prepared} --receptor-table ${curated}/receptor.tsv \\
        --config stage_config.json --output-dir ${prefix}/features/morgan
    """
}
