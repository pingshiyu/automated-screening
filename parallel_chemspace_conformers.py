from dg_functions import chemspace_root, chemspace_paths, evaluate_conformation, isomer_paths

# chemistry toolkit
from rdkit import Chem
from rdkit.Chem import AllChem

# standard libraries
import numpy as np
import pickle
import time, logging, sys

import multiprocessing as mp
from multiprocessing import Lock, Value

logging.basicConfig(filename='logs/conformers.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def initialize_counter(counter):
    global mol_counter
    mol_counter = counter

def mol_3d_err(idx, mol, attempts=3, use_counter=True):
    '''
    Parallelizable unit function: evaluate the loss function of a single molecule.
    If the molecule does not exist: loss is None;
    If the embedding failed: loss = -1;
    If the embedding was successsful, return the losses, currently w.r.t. the
        bond lengths and the bond angles (via `evaluate_conformation`, returning a
        list of [bond_err, angle_err]).
    Input:
        idx: identifier of molecule in the list
        mol: rdkit molecule object
    '''
    if use_counter:
        with mol_counter.get_lock():
            mol_counter.value += 1
            print(f'Processed {mol_counter.value} molecules \t\t\t', end='\r')
    else:
        print(f'Processing molecule #{idx} \t\t\t', end='\r')

    if not mol:
        return [idx, None]

    # prepare molecule: adding H improves conformation
    mol_with_h = Chem.AddHs(mol)

    # attempt to apply DG to embed molecule `attempts` times
    j = 0
    while j < attempts:
        try:
            success = AllChem.EmbedMolecule(mol_with_h)
            if success != -1:
                error = evaluate_conformation(mol_with_h)
                return [idx, error]
            # failure:
            j += 1
        except Exception as e:
            logging.error(f'Problem with molecule #{idx}: {Chem.MolToSmiles(mol)}')
            #logging.error(f'Here is the error message: {e.message}')
            logging.info(f'Molecule was skipped with error {e}.')
            # conformation process failed
            return [idx, None]

    # could not find conformation for the molecule
    return [idx, -1]

def parallel_mols_dg_errors(chemspace, attempts=3, threads=None, verbose=True, use_counter=True):
    '''
    Parallel version of `mols_dg_errors` function from `dg_functions`.
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
            for output in pool.starmap(mol_3d_err, input_args):
                results.append(output)
        except Exception as e:
            raise Exception(e.message)
    t1 = time.time()
    logging.info(f'Conformations completed. Elapsed time: {t1 - t0}.')

    if results:
        reading_failed = [idx for idx, loss in results if loss is None]
        conformation_failed = [idx for idx, loss in results if loss is -1]
        losses = np.array([[idx] + loss for idx, loss in results
                                if idx not in reading_failed and
                                   idx not in conformation_failed])
        if len(losses) > 0:
            sorted_ids = losses[:, 1].argsort()[::-1]
            losses = losses[sorted_ids]

        if verbose:
            logging.info(f'Worst conformations:\t\t\t\n {losses[:10]}')
            logging.info(f'Some {len(reading_failed)} / {len(chemspace)} molecules could not be read in.')
            logging.info(('Of these, '
                  f'{len(conformation_failed)} / {len(chemspace) - len(reading_failed)} '
                   'molecules failed to have a conformation.'))
            logging.info(f'Examples include: {conformation_failed[:10]}\n')

        return losses, reading_failed, conformation_failed
    else:
        return None, None, None

if __name__ == '__main__':
    chemspaces_to_calculate = ['c6h6', '125_56k', 'gdb_50k', 'pubchem_45k', 'chembl_50k', '125_338k']

    for name in chemspaces_to_calculate:
        logging.info(f'Computing chemspace: {name}')
        chemspace = Chem.SmilesMolSupplier(chemspace_paths[name], delimiter='\t')
        losses, reading_failed, conformation_failed = parallel_mols_dg_errors(chemspace, threads=8)

        with open(f'./output/avg_distance_error_bonds_avg_angle_error_by_degree_uff/{name}', 'wb') as f:
            pickle.dump([losses, reading_failed, conformation_failed], f)
