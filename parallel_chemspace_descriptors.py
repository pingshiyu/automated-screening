from dg_functions import chemspace_paths
from indices import all_indices, all_feature_vectors

descriptor_names, descriptors = all_indices()
feature_vector_names, feature_vectors = all_feature_vectors()
features_names = descriptor_names + feature_vector_names

# Chemistry libraries
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys

# standard libraries
import numpy as np
import pickle
import time, logging, sys

import multiprocessing as mp
from multiprocessing import Lock, Value

logging.basicConfig(filename='logs/descriptors.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def initialize_counter(counter):
    global mol_counter
    mol_counter = counter

def calc_feature_vectors(mol, feature_vectors):
    '''
    Compute all the descriptors which outputs a vector `feature_vectors` for
        a molecule `mol`.
    Returns a list of all of the entries computed
    Input:
        mol: RDKit molecule
        feature_vectors: list of functions, taking in an RDKit molecule and
            outputting a vector of values
    '''
    return np.concatenate([feature_vector(mol)
                           for feature_vector in feature_vectors])

def calc_descriptors(mol, descs):
    '''
    For a given molecule, compute all of the descriptors in `descs`.
    Returns a list of doubles (which can be null when the descriptors are
        undefined for that molecule)
    Input:
        mol: RDKit molecule
        descs: list of functions, taking in an RDKit molecule as first argument.
    '''
    return [desc(mol) for desc in descs]
    
def calc_all_features(idx, mol, use_counter=True, descs=descriptors):
    '''
    Parallelizable function to calculate all 2-D features of a molecule using
    RDKit. This includes topological indices/descriptors and fingerprints
    Returns a tuple (molecule ID, np array consisting of the features)
    Input:
        mol: RDKit molecule
        descs: list of functions, taking in an RDKit molecule as first argument.
    '''
    if use_counter:
        with mol_counter.get_lock():
            mol_counter.value += 1
            print(f'Processed {mol_counter.value} molecules \t\t\t', end='\r')
    else:
        print(f'Processing molecule #{idx} \t\t\t', end='\r')

    if mol:
        features = np.concatenate((calc_descriptors(mol, descs),
                                   calc_feature_vectors(mol, feature_vectors)))
        # also include the index to keep track of which molecule this is
        return idx, features

    # molecule couldn't be read:
    return idx, None

def parallel_mols_features(chemspace, threads=None, use_counter=True, verbose=True):
    # Parallelized molecule conformation runs:
    if not threads: # no threads defined
        threads = mp.cpu_count() - 1

    t0 = time.time()
    counter = Value('i', 0)
    with mp.Pool(initializer=initialize_counter, initargs=(counter,), processes=threads) as pool:
        logging.info(f'Using {threads} threads.')

        input_args = ((idx, mol, use_counter) for idx, mol in enumerate(chemspace))
        output = []
        try:
            for result in pool.starmap(calc_all_features, input_args):
                output.append(result)
        except Exception as e:
            logging.info('Error encountered:')
            logging.error(e.message)
            raise Exception(e.message)
    t1 = time.time()
    logging.info(f'Elapsed time: {t1 - t0}.\n')

    invalid_mols = [idx for idx, features in output
                    if features is None]
    # merge the id and the descriptors into one array
    mol_features = [np.concatenate(([idx], features)) for idx, features in output
                    if not (features is None)]

    return np.array(mol_features), invalid_mols

if __name__ == '__main__':
    chemspaces_to_calculate = ['c5h6']#['c6h6', 'gdb_50k', 'gdb_100k', '125_56k', '125_113k', '125_338k', 'pubchem_45k', 'pubchem_90k', 'chembl_50k', 'chembl_100k']

    for name in chemspaces_to_calculate:
        logging.info(f'Computing chemspace: {name}')
        chemspace = Chem.SmilesMolSupplier(chemspace_paths[name], delimiter='\t')
        features, invalids = parallel_mols_features(chemspace)

        with open(f'./output/descriptors_and_maccs/{name}_descriptors', 'wb') as f:
            pickle.dump([features_names, features, invalids], f)
