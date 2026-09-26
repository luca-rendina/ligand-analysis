// Split, model, prediction and report stages (ligand-ml image). Labels are joined here only.

process SPLIT {
    label 'ml'
    publishDir "${params.outdir}/split", mode: 'copy'

    input:
    path curated
    path ligands
    val config_json

    output:
    path 'split.tsv'

    script:
    """
    echo '${config_json}' > stage_config.json
    ligand-analysis split ${curated}/candidates.tsv ${ligands} --config stage_config.json --output split.tsv
    """
}

process TRAIN {
    tag "${representation}/${model}"
    label 'ml'
    publishDir "${params.outdir}/models", mode: 'copy'

    input:
    tuple val(representation), path(features), val(model)
    path cohort, stageAs: 'cohort/*'
    path split
    path curated
    val config_json

    output:
    tuple val(representation), val(model), path("${representation}_${model}")

    script:
    """
    echo '${config_json}' > stage_config.json
    ligand-analysis train --features ${features} --split ${split} --labels ${curated}/candidates.tsv \\
        --cohort ${cohort} --model ${model} --config stage_config.json --evaluate \\
        --output-dir ${representation}_${model}
    """
}

process PREDICT {
    tag "${representation}/${model}"
    label 'ml'
    publishDir "${params.outdir}/prediction/predictions", mode: 'copy'

    input:
    tuple val(representation), val(model), path(model_dir), path(features, stageAs: 'features')

    output:
    path "${representation}_${model}.tsv"

    script:
    """
    ligand-analysis predict --model-dir ${model_dir} --features features --output ${representation}_${model}.tsv
    """
}

process REPORT {
    label 'ml'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path curated
    path ligands
    path receptor
    path prepared, stageAs: 'prepared'
    path docking, stageAs: 'docking/*'
    path poses
    path redocking
    path features, stageAs: 'features/*'
    path split
    path models, stageAs: 'models/*'
    path predictions, stageAs: 'predictions/*'
    path run_metadata

    output:
    path 'report'

    script:
    """
    ligand-analysis report --curated ${curated} --ligand-inputs ${ligands} --receptor-dir ${receptor} \\
        --ligands-dir prepared --docking-dir ${docking} --poses-dir ${poses} --redocking-dir ${redocking} \\
        --features ${features} --split ${split} --model-dir ${models} --predictions ${predictions} \\
        --run-metadata ${run_metadata} --output-dir report
    """
}
