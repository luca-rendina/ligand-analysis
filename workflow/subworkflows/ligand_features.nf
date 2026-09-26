// Preparation, docking, pose selection and featurization of one ligand input table.
// Used for the labelled demo ligands and for unlabeled prediction ligands alike.

include { PREPARE_LIGANDS ; DOCK ; SELECT_POSES ; FEATURIZE_PLEC ; FEATURIZE_MORGAN } from '../modules/chem'

workflow LIGAND_FEATURES {
    take:
    prefix      // output folder: 'labelled' or 'prediction'
    ligands     // identity-only ligand input table
    receptor    // prepare-receptor output
    curated     // curation output (receptor.tsv gives the sample receptor ID)
    configs     // map of stage configuration JSON strings

    main:
    PREPARE_LIGANDS(prefix, ligands, configs.ligand_preparation)
    def jobs = ligands
        .splitCsv(header: true, sep: '\t')
        .map { row -> [prefix, row.ligand_id, row.ligand_id.replaceAll('[^A-Za-z0-9._-]', '_')] }
    DOCK(jobs, receptor, PREPARE_LIGANDS.out, configs.docking)
    SELECT_POSES(prefix, ligands, PREPARE_LIGANDS.out, DOCK.out.collect(), receptor, configs.pose_selection)
    FEATURIZE_PLEC(prefix, SELECT_POSES.out, receptor, configs.featurization)
    FEATURIZE_MORGAN(prefix, PREPARE_LIGANDS.out, curated, configs.featurization)

    emit:
    prepared = PREPARE_LIGANDS.out
    docking = DOCK.out.collect()
    poses = SELECT_POSES.out
    features = FEATURIZE_PLEC.out.mix(FEATURIZE_MORGAN.out)
}
