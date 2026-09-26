/*
 * ligand-analysis public demo, from acquisition to report:
 * fetch -> curate -> ligand inputs -> prepare receptor/ligands -> dock -> select poses
 * -> PLEC and Morgan features -> persisted split -> models -> evaluation -> unlabeled
 * prediction -> HTML report. Scientific parameters come from the params file
 * (configs/demo.yaml); execution settings from nextflow.config and conf/.
 *
 *   nextflow run main.nf -profile podman -params-file configs/demo.yaml
 *
 * Each stage receives only its configuration sections, so -resume reruns a stage (and what
 * depends on its outputs) only when those sections or its inputs change.
 */

include { FETCH ; CURATE ; LIGAND_INPUTS ; PREDICTION_INPUTS } from './workflow/modules/data'
include { PREPARE_RECEPTOR ; REDOCK_REFERENCE } from './workflow/modules/chem'
include { SPLIT ; TRAIN ; PREDICT ; REPORT ; CHECK_REPORT } from './workflow/modules/ml'
include { LIGAND_FEATURES as LABELLED ; LIGAND_FEATURES as PREDICTION } from './workflow/subworkflows/ligand_features'

def stageConfig(sections) {
    def missing = sections.findAll { name -> !params.containsKey(name) }
    if (missing) {
        error("The params file lacks the section(s): ${missing.join(', ')}")
    }
    def json = groovy.json.JsonOutput.toJson(sections.collectEntries { name -> [(name): params[name]] })
    if (json.contains("'")) {
        error("Configuration values must not contain single quotes (sections ${sections})")
    }
    return json
}

// Image ID of a container tag for the run record (null when no Podman client is available).
def imageId(name) {
    try {
        def proc = ['podman', 'image', 'inspect', '--format', '{{.Id}}', name].execute()
        def out = proc.in.text.trim()
        proc.waitFor()
        return proc.exitValue() == 0 ? out : null
    }
    catch (Exception _error) {
        return null
    }
}

workflow {
    main:
    def manifestFile = file(params.manifest, checkIfExists: true)
    def manifest = new org.yaml.snakeyaml.Yaml().load(manifestFile.text)
    // Values that become file names or command arguments are checked before any task runs;
    // the stage commands validate everything else against the JSON schemas.
    if (!(manifest.name ==~ /[a-z0-9][a-z0-9_-]*/)) {
        error("Manifest name must match [a-z0-9][a-z0-9_-]*: ${manifest.name}")
    }
    def classifiers = ['dummy', 'logistic_regression', 'random_forest', 'legacy_ensemble']
    def unknownClassifiers = params.models.classifiers.findAll { name -> !(name in classifiers) }
    if (unknownClassifiers) {
        error("Unknown classifiers ${unknownClassifiers}; choose from ${classifiers}")
    }
    def configs = [
        ligand_preparation: stageConfig(['ligand_preparation']),
        docking: stageConfig(['docking']),
        pose_selection: stageConfig(['pose_selection']),
        featurization: stageConfig(['featurization']),
    ]

    FETCH(manifestFile, manifest.name)
    CURATE(manifestFile, FETCH.out)
    LIGAND_INPUTS(CURATE.out, stageConfig(['ligand_inputs']))
    PREPARE_RECEPTOR(manifestFile, FETCH.out, stageConfig(['receptor']))
    REDOCK_REFERENCE(PREPARE_RECEPTOR.out, stageConfig(['ligand_preparation', 'docking', 'redocking', 'pose_selection']))

    LABELLED('labelled', LIGAND_INPUTS.out, PREPARE_RECEPTOR.out, CURATE.out, configs)
    SPLIT(CURATE.out, LIGAND_INPUTS.out, stageConfig(['split']))
    def featureDirs = LABELLED.out.features.map { entry -> entry[1] }.collect()
    TRAIN(
        LABELLED.out.features.combine(channel.fromList(params.models.classifiers)),
        featureDirs, SPLIT.out, CURATE.out, stageConfig(['models'])
    )

    def predictions = channel.value([])
    if (params.prediction.ligand_ids) {
        PREDICTION_INPUTS(CURATE.out, stageConfig(['prediction']))
        PREDICTION('prediction', PREDICTION_INPUTS.out, PREPARE_RECEPTOR.out, CURATE.out, configs)
        PREDICT(TRAIN.out.combine(PREDICTION.out.features, by: 0))
        predictions = PREDICT.out.collect()
    }

    def runMetadata = channel.value(groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson([
        name: params.name,
        run_name: workflow.runName,
        session_id: workflow.sessionId.toString(),
        resumed: workflow.resume,
        nextflow_version: nextflow.version.toString(),
        command_line: workflow.commandLine,
        profile: workflow.profile,
        launch_dir: workflow.launchDir.toString(),
        work_dir: workflow.workDir.toString(),
        start: workflow.start.toString(),
        git_revision: params.git_revision,
        containers: params.containers.collectEntries { key, image -> [(key): [image: image, id: imageId(image)]] },
        config_file: workflow.commandLine.find('-params-file\\s+\\S+'),
        params: params,
    ]))).collectFile(name: 'run_metadata.json', newLine: true)

    REPORT(
        CURATE.out, LIGAND_INPUTS.out, PREPARE_RECEPTOR.out, LABELLED.out.prepared, LABELLED.out.docking,
        LABELLED.out.poses, REDOCK_REFERENCE.out, featureDirs, SPLIT.out,
        TRAIN.out.map { entry -> entry[2] }.collect(), predictions, runMetadata
    )
    CHECK_REPORT(REPORT.out)

    onComplete:
    log.info("Run ${workflow.runName} ${workflow.success ? 'succeeded' : 'failed'}; report: ${params.outdir}/report/report.html")
}
