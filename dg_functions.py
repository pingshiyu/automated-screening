# rdkit aliases
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType
bond_symbol = {
    BondType.SINGLE: '-',
    BondType.AROMATIC: '+',
    BondType.DOUBLE: '=',
    BondType.TRIPLE: '#',
}

import math, numpy as np
import multiprocessing as mp
import pickle
from statistics import mean
import bisect # for histograms

# used in `get_conformer_angles`
from itertools import combinations

chemspace_root = './molecules/'
chemspace_files = {
    '125_338k': 'chemspace_125_338k.smi',
    '125_113k': 'chemspace_125_113k.smi',
    '125_56k': 'chemspace_125_56k.smi',
    'gdb_50k': 'gdb17_sample_50k.smi',
    'gdb_100k': 'gdb17_sample_100k.smi',
    'chembl_50k': 'chembl_50k.smi',
    'chembl_100k': 'chembl_100k.smi',
    'pubchem_45k': 'pubchem_45k.smi',
    'pubchem_90k': 'pubchem_90k.smi',
    'c6h6': 'C6H6.smi',
    'c5h6': 'C5H6.smi',
}
chemspace_paths = {n: f'{chemspace_root}{fp}' for n, fp in chemspace_files.items()}

# Earlier cached outputs from parallel_chemspace_angles.py -> parallel_mol_angles()
angles_files = ['chembl_50k', 'gdb_50k', 'pubchem_45k']
angles_root = './output/angles/'
angles_paths = {name: f'{angles_root}{name}' for name in angles_files}

# Earlier cached outputs of real angle distributions (on common bond types)
real_cos_angles_distribution_path = angles_root + 'real_cos_angles_distribution'
with open(real_cos_angles_distribution_path, 'rb') as f:
    real_cos_angles_distribution = pickle.load(f)
    #print(f'Real angles distribution successfully loaded!')

def sq_error_normalised(dist, lb, ub):
    '''
    Evaluate the errors of a distance, given the bounds `lb`, `ub`, and `dist`
    Formula \sum_{bonds} max((lb - actual)/lb, 0)^2 + max((actual - ub)/ub, 0)^2
    '''
    if (dist < lb):
        return ((lb**2 - dist**2) / lb**2)**2
    elif (dist > ub):
        return ((dist**2 - ub**2) / ub**2)**2
    else:
        return 0

def distance_error_bonds(mol, dist_error_fn=sq_error_normalised):
    '''
    Evaluate the distance-based errors of the molecule (on bonds only)
    Outputs the distance errors
    '''
    dist_matrix = Chem.rdmolops.Get3DDistanceMatrix(mol)
    bounds_matrix = AllChem.GetMoleculeBoundsMatrix(mol)

    error_terms = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        i, j = min(begin, end), max(begin, end) # so that i < j

        # lower bound in lower triangle, upper bound in upper triangle.
        lb, ub = bounds_matrix[j, i], bounds_matrix[i, j]
        dist = dist_matrix[i, j]

        error_terms.append(dist_error_fn(dist, lb, ub))

    return error_terms

def distance_error_pairs(mol, dist_error_fn=sq_error_normalised):
    '''
    Evaluate the distance-based errors of the molecule (on all atom pairs), based
    on the function `dist_error_fn`
    Outputs the distance errors
    '''
    dist_matrix = Chem.rdmolops.Get3DDistanceMatrix(mol)
    bounds_matrix = AllChem.GetMoleculeBoundsMatrix(mol)

    n_atoms = mol.GetNumAtoms()

    error_terms = []
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            assert(i < j)
            # lower bound in lower triangle (j, i), upper bound in upper triangle (i, j)
            lb, ub = bounds_matrix[j, i], bounds_matrix[i, j]
            dist = dist_matrix[i, j]

            error_terms.append(dist_error_fn(dist, lb, ub))

    return error_terms

def get_conformer_angles(conformer, logger=None):
    '''
    Given an RDKit Conformer `conformer`, compute all angles of the atoms (angles
    defined between any 3 adjacent atoms). The angles are stored in a dictionary of
    form:
    {
        'Ax=Ac-Ay': [angle1, angle2, ...],
        ...
    }
    where `Ac` is the central atom, `Ax`, `Ay` are two atoms bonded to `Ac`, with
    `Ax <= Ay` lexicographically; '-', '+', '=', '#' denotes single, aromatic,
    double, and triple bonds, connected to `Ac`, respectively.

    This function returns the dictionary in a different form vs. the one used in
    `parallel_chemspace_angles.py`.

    Input:
        conformer: RDKit Conformer object
        logger: logging.getLogger() object, provides a logging stream for errors
    Output:
        dictionary, angles dictionary object with keys of the bond type, and the
            values the list of angles present on the conformer (in radians)
    '''
    # get the molecule of this conformer
    mol = conformer.GetOwningMol()

    angles = {}
    for ac in mol.GetAtoms():
        neighbours = list(ac.GetNeighbors())
        if len(neighbours) <= 1: # only 1 neighbour: cannot have unambiguous angles
            continue

        Ac, ac_id, ac_deg = ac.GetSymbol(), ac.GetIdx(), ac.GetDegree()

        # find angles with `Ac` as the central atom
        for a1, a2 in combinations(neighbours, 2):
            A1, A2 = a1.GetSymbol(), a2.GetSymbol()
            # impose a lexicographic restriction on the atoms in the dictionary
            ax, ay = (a1, a2) if A1 <= A2 else (a2, a1)
            Ax, Ay = ax.GetSymbol(), ay.GetSymbol()

            # calculate bond angles
            ax_id, ay_id = ax.GetIdx(), ay.GetIdx()
            angle = Chem.rdMolTransforms.GetAngleRad(conformer, ax_id, ac_id, ay_id)

            # find out bond types
            bx, by = [bond_symbol.get(mol.GetBondBetweenAtoms(a_id, ac_id).GetBondType())
                      for a_id in [ax_id, ay_id]]
            if (not bx) or (not by):
                # undefined bond error, ignore current bound
                error_str = f'Problem in molecule: an {Ax, Ac, ac_deg, Ay} bond was ignored.'
                if not logger:
                    print(error_str)
                else:
                    logger.error(error_str)
                continue

            # write as a dictionary entry
            entry = f'{Ax}{bx}{Ac}{ac_deg}{by}{Ay}'
            if entry not in angles:
                angles[entry] = [angle]
            else:
                angles[entry].append(angle)
    return angles

def _angle_error_nllf(bond_type, bond_angles, real_cos_angles_distribution, max_err=10.0):
    '''
    Computes the negative log likelihood function for the `bond_type`, with angles
    `bond_angles` based on the empirical bond cosine-angle distribution
    `real_cos_angles_distribution`
    Returns an list of the associated 'loss' with each angle in `bond_angles`.
    Input:
        bond_type: string, representing the type of the bond ("AxbxAcbyAy"): central
            atom 'Ac' with adjacent atoms 'Ax' and 'Ay'
        bond_angles: list of double, observed angles (in radians)
    '''
    #print(f'{bond_type} with angles: {bond_angles}')
    bond_angle_pmf = real_cos_angles_distribution.get(bond_type)
    # if the bond type was not included in the real molecules angle distribution
    # (i.e. due to lack of examples), use the default probability
    if not bond_angle_pmf:
        #print(f'Instances of {bond_type} are too rare to have reliable distribution: default probability used\n')
        return [-math.log(real_cos_angles_distribution['default'])] * len(bond_angles)
    else:
        pmf, bins = bond_angle_pmf
        losses = []
        for angle in bond_angles:
            # `bisect_left(sections, x)` returns the index in `sections` which `x` belong in
            p_of_cos_angle = pmf[bisect.bisect_left(bins, math.cos(angle)) - 1]
            #print(f'cos(angle)={math.cos(angle)}, p(cos(angle))={p_of_cos_angle}, \nbins={bins}, \npmf={pmf}\n')
            loss = max_err if p_of_cos_angle == 0 else -math.log(p_of_cos_angle, 10)
            losses.append(loss)
        return losses

def nllf_all_angles(mol,
                     real_cos_angles_distribution=real_cos_angles_distribution,
                     max_err=10.0):
    '''
    Evaluate the error w.r.t. the bond angles (on all 3 adjacent atoms), by
    computing by the negative log-loss to the `real_cos_angles_distribution`
    object (a `scipy.rv_continuous` distribution which is the angles distribution
    observed in real molecules)
    Outputs the angle errors

    Inputs:
        conformer: RDKit molecule, with a conformer succesfully calculated
        real_cos_angles_distribution: dictionary mapping bond types to
            scipy.rv_continuous distributions, observed CHNOPS angles distribution
            of real molecules (e.g. ChEMBL, PubChem). The distribution pmf is
            defined as a histogram, see `angles-analysis.ipynb` for details.
            Used to evaluate the deviation from realism in the bond angles.
        max_err: value cap for an evaluated error - we use log losses, which means
            `infty` errors are possible without a cap
    '''
    # Get all angles within the conformer currently
    conformer_angles = get_conformer_angles(mol.GetConformer())

    error_terms = []
    for bond_type, bond_angles in conformer_angles.items():
        error_terms += _angle_error_nllf(bond_type, bond_angles,
                                        real_cos_angles_distribution,
                                        max_err=max_err)
    return error_terms

def uff_energy(mol):
    '''
    Computes the average potential energy (per atom), of the molecule, via the
    UFF (universal forcefield).
    '''
    # attmpt to initialize a forcefield - this may fail due to, e.g. exotic atom
    # bonding. This results in a RuntimeError
    try:
        ff = Chem.rdForceFieldHelpers.UFFGetMoleculeForceField(mol)
    except RuntimeError as e:
        return None

    ff.Initialize()
    ff.Minimize()
    energy = ff.CalcEnergy() / mol.GetNumAtoms()

    return energy

def evaluate_conformation(mol,
                          dist_error_fn=distance_error_bonds,
                          dist_agg=mean,
                          angle_error_fn=nllf_all_angles,
                          angle_agg=mean):
    '''
    Evaluate the accuracy of the conformation obtained of the molecule, based
    on the observed 'realistic' bond angles and 'realistic' bond distances.
    Outputs the average errors computed on the bond lengths, and bond angles

    The molecule is expected to have at least 1 conformer already found. That
    is, mol.GetConformer() should return (and not throw).

    Input:
        mol: molecule to be evaluated
        dist_error_fn: function to evaluate the bond lengths of a conformer of
            the `mol`. Takes a RDKit `Mol` object as the first argument.
        angle_error_fn: function to evaluate the angles on a conformer. Takes a
            RDKit `Conformer` object as its first argument.
        dist_agg, angle_agg: functions taking in array-like inputs, used for
            aggregating together the distance and angle error terms respectively
    '''
    # Distance error terms: deviation from the ideal bond lengths for each bond
    distance_error_terms = dist_error_fn(mol)
    distance_error = dist_agg(distance_error_terms)

    # Angle error terms: for each 3 adjacent atoms, deviation from the observed
    # bond angle.
    angle_error_terms = angle_error_fn(mol)
    angle_error = angle_agg(angle_error_terms)

    # Forcefield potential energy (should compute this last, as this may change
    # the conformer geometry)
    uff_e = uff_energy(mol)

    return [distance_error, angle_error, uff_e]

def mols_dg_errors(chemspace, retry_failed=True, retry_attempts=3):
    '''
    Given a set of molecules,
        1. generate 3D conformers using DG
        2. evaluate conformers according to error function
        3. pick out molecules that failed to conform
    Some molecules may fail to have a conformation the first time round: we will
    retry `retry_attempts` with these. If they still fail then they really are
    utter failures.

    Input:
        chemspace: list of molecules
    Output:
        np array, of entries [id, error], id are the ids of `chemspace`
        list, ids of `chemspace` that failed to have a conformation
    '''
    failures = []
    errors = []

    # attempt to optimize once initially
    for i, mol in enumerate(chemspace):
        print(f'Reading molecule #{i}', end='\r')

        if not mol:
            continue

        mol_with_h = Chem.AddHs(mol)
        failure = AllChem.EmbedMolecule(mol_with_h)
        if failure == -1:
            failures.append(i)
            continue

        error = evaluate_conformation(mol_with_h)
        errors.append([i, error])
    print(f'Initially: {len(failures)}/{len(chemspace)} molecules failed conformation')

    # for failed molecules, retry `retry_attempts` times
    if retry_failed:
        true_failures = []
        for i in failures:
            failed_mol = chemspace[i]
            failed_mol_with_h = Chem.AddHs(failed_mol)

            failed_n_times = True
            j = 0
            while failed_n_times and j < 3:
                failure = AllChem.EmbedMolecule(failed_mol_with_h)
                # failed again:
                if failure == -1:
                    j += 1
                    continue

                # success:
                failed_n_times = False
                error = evaluate_conformation(mol_with_h)
                errors.append([i, error])

            if failed_n_times:
                true_failures.append(i)

        print(f'Of these, {len(true_failures)}/{len(failures)} still failed after another {retry_attempts} tries.')

    # sort the errors, largest to smallest, before returning
    if errors:
        errors = np.array(errors)
        sorted_ids = errors[:, 1].argsort()[::-1]
        errors = errors[sorted_ids]

    if retry_failed:
        return errors, true_failures
    else:
        return errors, failures

def merge_angles_dicts(angles1, angles2):
    '''
    'Deep merge' of two dictionaries containing angles information together.
    `angles1`, `angles2` are nested dictionaries indexed by:
        1. the central atom [`Ac`]
        2. the 2 atoms connected to `Ac`: `Ax`, `Ay` & respective bonds `bx`, `by`
            connected to the central atom [(`Axbx`, `byAy`)]
    with values being the list of angles observed between `AxbxAcbyAy` bond type.

    For use in a reduce function merging chains of dictionaries together.
    This is the angles format used by `mol_3d_angles` in `parallel_chamspace_angles.py`
    and is the main user of this function.

    Inputs:
        angles1, angles2: angles dictionaries in the aforementioned format
    Output:
        a deep-merged dictionary of angles1 and angles2.
    '''
    merged = angles1
    for Ac, angles_Ac in angles2.items():
        if Ac not in merged:
            merged[Ac] = angles_Ac
        else: # need to merge together
            for neighbours, angles_list in angles_Ac.items():
                if neighbours not in merged[Ac]:
                    merged[Ac][neighbours] = angles_list
                else:
                    merged[Ac][neighbours] += angles_list
    return merged
