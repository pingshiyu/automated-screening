from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors as MD, GraphDescriptors as GD, MACCSkeys
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors

import numpy as np

def wiener(mol):
    '''
    Compute the Wiener index of `mol`.
    This is the upper-triangular sum of the molecule's distance matrix.
    '''
    distance_matrix = Chem.rdmolops.GetDistanceMatrix(mol)
    upper_triangle = [row[i+1:] for i, row in enumerate(distance_matrix)]
    return sum([np.sum(arr) for arr in upper_triangle])

def crowding(mol, d):
    '''
    Compute the distance-d crowding index of `mol`.
    '''
    distance_matrix = Chem.rdmolops.GetDistanceMatrix(mol)
    neighbourhood_sizes = [np.sum(v_distances<=d) for v_distances in distance_matrix]
    return max(neighbourhood_sizes)

def crowding_functions(low=2, high=4):
    '''
    Returns a list of functions which takes in a molecule and calculates the crowding index
    at different values of `d`.
    Input:
        low, high: lower / upper bound for `d`, inclusive.
    '''
    return [lambda mol, d=i: crowding(mol, d) for i in range(low, high+1)]

def all_indices(crowding_low=2, crowding_high=4):
    # Get default descriptors of RDKit
    descriptor_names, descriptors = zip(*MD.descList)
    descriptor_names, descriptors = list(descriptor_names), list(descriptors)

    # get missing (new) descriptors in RDKit
    # patching remaining descriptors not in all_indices
    extra_descs = [
        ('NumAmideBonds', rdMolDescriptors.CalcNumAmideBonds),
        #('NumAtomStereoCenters', rdMolDescriptors.CalcNumAtomStereoCenters),
        ('NumBridgeheadAtoms', rdMolDescriptors.CalcNumBridgeheadAtoms),
        ('NumHBA', rdMolDescriptors.CalcNumHBA),
        ('NumHBD', rdMolDescriptors.CalcNumHBD),
        ('NumLipinskiHBA', rdMolDescriptors.CalcNumLipinskiHBA),
        ('NumLipinskiHBD', rdMolDescriptors.CalcNumLipinskiHBD),
        ('NumSpiroAtoms', rdMolDescriptors.CalcNumSpiroAtoms),
        #('NumUnspecifiedAtomStereoCenters', rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters)
    ]
    extra_descs_names, extra_descs = zip(*extra_descs)
    descriptor_names += extra_descs_names
    descriptors += extra_descs

    # Adding extra descriptors:
    # distance based descriptors: the Wiener Index and the Crowding Index
    crowding_names = [f'crowding_{i}' for i in range(crowding_low, crowding_high+1)]
    crowding_fns = crowding_functions(low=crowding_low, high=crowding_high)
    descriptor_names += crowding_names
    descriptors += crowding_fns

    # Wiener index:
    descriptor_names += ['wiener']
    descriptors += [wiener]

    return descriptor_names, descriptors

def calc_maccs_fingerprint(mol):
    '''
    Calculate the MACCS fingerprint for the molecule `mol` and return as a
    feature vector
    '''
    result = np.empty(167)
    fp = MACCSkeys.GenMACCSKeys(mol)
    DataStructs.cDataStructs.ConvertToNumpyArray(fp, result)
    return result

def all_feature_vectors():
    '''
    Returns all feature vectors: a feature vector will output more than just a
        single value. A feature_vector is a function taking in an RDKit molecule,
        and outputting a vector of real values.
    Returns a tuple of feature_names, feature_vectors where feature_names is a
        list of size \sum_{feature in feature_vectors} (dim(Im(feature))), i.e.
        sum of all feature vector's lengths, in order.
    '''
    feature_names, feature_vectors = [], []

    # MACCS fingerprints: used as a feature vector
    feature_names += [f'MACCS_{i}' for i in range(167)]
    feature_vectors += [calc_maccs_fingerprint]

    # autocorr2d features
    feature_names += [f'AUTOCORR2D_{i}' for i in range(192)]
    feature_vectors += [rdMolDescriptors.CalcAUTOCORR2D]

    '''
    These descriptors are unusable unfortunately as the mapped vectors are not
    of a fixed length
    # EState descriptors
    feature_names += [f'EState_{i}' for i in range(6)]
    feature_vectors += [Chem.EState.EStateIndices]

    # Feature invariants
    feature_names += [f'Invariant_{i}' for i in range(6)]
    feature_vectors += [rdMolDescriptors.GetFeatureInvariants]
    '''

    return feature_names, feature_vectors
