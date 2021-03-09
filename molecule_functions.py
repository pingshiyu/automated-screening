import numpy as np, pandas as pd

# chemistry libraries
from openbabel import openbabel as ob, pybel

def bernoulli_generator(p, batch=10000):
    '''
        Returns a Ber(p) distribution generator. P(B == 1) = p
        Input:
            p: float, real number, p in interval [0, 1]
            batch: int, number of samples to generate each time
    '''
    while True:
        # generate `batch` samples at a time (for better performance)
        samples_batch = np.random.binomial(1, p, size=batch)
        for sample in samples_batch:
            yield sample
            
def _get_directed_bond_length_range(dist, a1, a2, order, method='exact'):
    '''
        Get the directed bond length range within `dist` database.
    '''
    if (method == 'exact'):
        return dist.loc[a1, a2, order]['avg'], dist.loc[a1, a2, order]['avg']
    elif (method == 'minmax'):
        return dist.loc[a1, a2, order]['min'], dist.loc[a1, a2, order]['max']
    else: # either (method == 'unc') or other input.
        unc = dist.loc[a1, a2, order]['unc']
        return dist.loc[a1, a2, order]['avg'] - unc, dist.loc[a1, a2, order]['max'] + unc
    
def _get_bond_length_range(dist, a1, a2, order, method='exact'):
    '''
        Get the bond length range (outputting (lb, ub) tuple) of `order`, between `a1` and `a2`
        within bond length database `dist`. Method used is detailed in molecule_to_dg.
    '''
    # read from the distance data
    if (a1, a2, order) in dist.index:
        return _get_directed_bond_length_range(dist, a1, a2, order, method=method)
    elif (a2, a1, order) in dist.index:
        return _get_directed_bond_length_range(dist, a2, a1, order, method=method)
    else:
        # fallback to the average bond length range between organic atoms if not present
        return 1.00, 1.50
            
def molecule_to_dg(mol, dist, method='exact'):
    '''
        Convert a pybel molecule to a dataframe object
        The output dataframe has columns:
            [atom_a, atom_b, dist_lower, dist_upper]
            with each row representing a bond.
        The distances are from `dist`, which contains the typical bond lengths for each type
        Inputs:
            mol: pybel molecule object
            dist: dataframe, containing the distances information in the columns for each atom type
                columns are: [a1, a2, ord, avg, min, max, unc], 
                indexed by columns [a1, a2, ord] 
                length units are in angstroms.
            method: string, will take options: 'exact', 'minmax', 'unc' representing:
                'exact': only the average distance
                'minmax': range of the shortest/longest observed bond
                'unc': within measurement uncertainty
    '''
    mol_dist_data = []
    for bond in ob.OBMolBondIter(mol.OBMol):
        # extract bond information
        a1, a2, order = bond.GetBeginAtom().GetType()[0], bond.GetEndAtom().GetType()[0], bond.GetBondOrder()
        a1_idx, a2_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Get distance range from dist matrix based on `method`
        min_dist, max_dist = _get_bond_length_range(dist, a2, a1, order, method=method)
        
        mol_dist_data.append([a1_idx, a2_idx, min_dist, max_dist])
        
    return pd.DataFrame(mol_dist_data)

def sdf_to_smi(sdf_path, output_path, sample=0.1):
    '''
        Convert an SDF file to SMI file collection for easy reading / writing
        and compact storage.
        Inputs:
            sdf_path, output_path: string, input / output paths
            sample: float, proportion of the SDF database to sample.
    '''
    sampler = bernoulli_generator(sample)
    output_writer = pybel.Outputfile('smi', output_path, overwrite=True)
    
    sample_size = 0
    for i, molecule in enumerate(pybel.readfile('sdf', sdf_path)):
        print(f'Progress: {i}', end='\r')
        if next(sampler):
            sample_size += 1
            output_writer.write(molecule)
    print(f'Converted {sample_size} molecules successfully and written to: {output_path}')