from dg_functions import chemspace_root, chemspace_paths, evaluate_conformation, isomer_paths
from dg_functions import merge_angles_dicts

# chemistry toolkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType
bond_symbol = {
    BondType.SINGLE: '-',
    BondType.AROMATIC: '+',
    BondType.DOUBLE: '=',
    BondType.TRIPLE: '#',
}

# standard libraries
import numpy as np
import pickle
import time, logging, sys

# mathematical and FP functions
from itertools import combinations
from functools import reduce

import multiprocessing as mp
from multiprocessing import Lock, Value

logging.basicConfig(filename='logs/angles.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def initialize_counter(counter):
    global mol_counter
    mol_counter = counter

def has_conformer(idx, mol, attempts=3):
    '''
    Generate a conformer of the molecule `mol`, giving up after `attempts` tries.
    If we gave up before finding a conformer, return None,
    else return True.
    '''
    for _ in range(attempts):
        try:
            success = AllChem.EmbedMolecule(mol)
            if success != -1:
                return True
        except Exception as e:
            # conformation process failed
            logging.error(f'Problem with molecule #{idx}: {Chem.MolToSmiles(mol)}')
            #logging.error(f'Here is the error message: {e.message}')
            logging.info('Molecule was skipped.')
            return False
    return False

def mol_3d_angles(idx, mol, attempts=3, use_counter=True):
    '''
    Parallelizable unit function: generate a conformer and calculate the angles
    on the molecule. The angles are returned as a dictionary, with structure:
    {
        'Ac': {
            ('Ax-', 'Ay='): [angle1, angle2, ...],
            ...
        },
        ...
    }
    where `Ac` is the central atom; `Ax`, `Ay` are the two atoms connected
    to the central atom; '-', '+', '=', '#' denotes single, double, triple, aromatic bonds,
    connected to `Ac`, respectively.

    If the molecule does not have a conformation, the angles dictionary is `None`;
    If the molecule has a conformation, return its angles dictionary.
    '''
    if use_counter:
        with mol_counter.get_lock():
            mol_counter.value += 1
            print(f'Processed {mol_counter.value} molecules \t\t\t', end='\r')
    else:
        print(f'Processing molecule #{idx} \t\t\t', end='\r')

    if not mol:
        return [idx, None]

    # generate molecule conformer
    mol_with_h = Chem.AddHs(mol)
    if not has_conformer(idx, mol_with_h):
        return [idx, -1]

    # successfully found a conformer: evaluate its angles
    angles = {}
    conformer = mol_with_h.GetConformer()
    for ac in mol_with_h.GetAtoms():
        neighbours = list(ac.GetNeighbors())
        if len(neighbours) <= 1: # only 1 neighbour: cannot have unambiguous angles
            continue

        Ac, ac_id, ac_deg = ac.GetSymbol(), ac.GetIdx(), ac.GetDegree()
        Ac_key = f'{Ac}{ac_deg}'
        if Ac not in angles:
            angles[Ac_key] = {}

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
            bx, by = [bond_symbol.get(mol_with_h.GetBondBetweenAtoms(a_id, ac_id).GetBondType())
                      for a_id in [ax_id, ay_id]]
            if (not bx) or (not by):
                # undefined bond error, ignore current bound
                logging.error(f'Problem with molecule #{idx}: an {Ax, Ac_key, Ay} bond was ignored.')
                continue

            # write as a dictionary entry
            entry = (f'{Ax}{bx}', f'{by}{Ay}')
            if entry not in angles[Ac_key]:
                angles[Ac_key][entry] = [angle]
            else:
                angles[Ac_key][entry].append(angle)
    return [idx, angles]

def parallel_mols_angles(chemspace, attempts=3, threads=None, verbose=True, use_counter=True):
    '''
    Parallelised version of `mol_3d_err` function.
    Outputs a tuple of:
        (losses, reading_failed, conformation_failed)
    representing the chemspace's DG conformer evaluations, failed molecules (either
        failed to be read or failed to have a conformer).
    '''
    # Parallelized molecule conformation runs:
    if not threads: # no threads defined
        threads = mp.cpu_count() - 1

    t0 = time.time()
    counter = Value('i', 0)
    with mp.Pool(initializer=initialize_counter, initargs=(counter,), processes=threads) as pool:
        logging.info(f'Using {threads} threads.')

        results = []
        input_args = ((idx, mol, attempts, use_counter) for idx, mol in enumerate(chemspace))
        try:
            for output in pool.starmap(mol_3d_angles, input_args):
                results.append(output)
        except Exception as e:
            raise Exception(e.message)
    t1 = time.time()
    logging.info(f'Conformations completed. Elapsed time: {t1 - t0}.')

    if results:
        reading_failed = [idx for idx, angles in results if angles is None]
        conformation_failed = [idx for idx, angles in results if angles is -1]
        good_angles = [angles for idx, angles in results
                        if idx not in reading_failed and
                           idx not in conformation_failed]

        if verbose:
            logging.info(f'Some {len(reading_failed)} / {len(chemspace)} molecules could not be read in.')
            logging.info(('Of these, '
                  f'{len(conformation_failed)} / {len(chemspace) - len(reading_failed)} '
                   'molecules failed to have a conformation.'))
            logging.info(f'Examples include: {conformation_failed[:10]}\n')

        if len(good_angles) > 0:
            all_angles = reduce(merge_angles_dicts, good_angles)
        else:
            all_angles = {}

        return all_angles, reading_failed, conformation_failed
    else:
        return None, None, None

if __name__ == '__main__':
    chemspaces_to_calculate = ['c6h6', 'gdb_50k', 'pubchem_45k', 'chembl_50k']

    for name in chemspaces_to_calculate:
        logging.info(f'Computing chemspace: {name}')
        chemspace = Chem.SmilesMolSupplier(chemspace_paths[name], delimiter='\t')
        angles, reading_failed, conformation_failed = parallel_mols_angles(chemspace)

        with open(f'./output/angles/{name}', 'wb') as f:
            pickle.dump([angles, reading_failed, conformation_failed], f)
