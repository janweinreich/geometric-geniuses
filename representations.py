import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid, KFold
from qml.representations import generate_fchl_acsf
from cMBDF import generate_mbdf
from dscribe.descriptors import SOAP
from ase import Atoms
from joblib import Parallel, delayed
import glob
from qml.representations import get_slatm_mbtypes, generate_slatm
from qstack import compound, spahm
from qml.representations import generate_bob
import pdb
import tempfile

# lookup here!
# https://github.com/janweinreich/FML
# https://github.com/janweinreich/FML/blob/master/activation/activation_oneout_cv_krr_mbdf_gzip.py


def pad_max_size(X, max_size):
    X = np.array(
        [np.pad(subarray, (0, max_size - len(subarray)), "constant") for subarray in X]
    )
    return X


def gen_all_bob(COORDINATES, NUC_CHARGES, size=158, asize = {"O":3, "C":7, "N":3, "H":16, "S":1}):
    X = []
    for i in tqdm(range(len(COORDINATES))):
        X.append(generate_bob(NUC_CHARGES[i], COORDINATES[i],atomtypes=[el for el in asize.keys()],  size=size, asize = asize))
    
    X= np.array(X)
    
    return X

def get_all_slatm(CHARGES, COORDINATES, local=True):
    X = np.array(
                [np.array(generate_slatm(COORDINATES[i], CHARGES[i], 
                                                    mbtypes=get_slatm_mbtypes([CHARGES[i]]), rcut=8.0,
                                                            local=local))
                for i in range(len(CHARGES))])
    return X


def get_single_fchl(charges, coordinates, unique_charges,pad=50):

    x = generate_fchl_acsf(charges, coordinates, gradients=False, pad=pad, elements=unique_charges)

    return x

def get_all_fchl(CHARGES, COORDINATES, unique_charges, pad=50):

    X = Parallel(n_jobs=-1)(delayed(get_single_fchl)(CHARGES[i], COORDINATES[i], unique_charges, pad) for i in tqdm(range(len(CHARGES))))
    X = np.array(X)
    return X

def max_element_counts(elements):
    # Initialize an empty dictionary to store the maximum counts
    max_counts = {}

    # Iterate over each sub-array in the main array
    for sub_array in elements:
        # Count the occurrences of each element in the sub-array
        unique, counts = np.unique(sub_array, return_counts=True)
        count_dict = dict(zip(unique, counts))

        # Compare the counts with the current maximum counts
        for element, count in count_dict.items():
            if element not in max_counts or count > max_counts[element]:
                max_counts[element] = count

    #find the maximum number of each element in the list of lists
    max_n = max([len(x) for x in elements])

    return max_counts, max_n


def gen_soap(crds, chgs, species = ['Br', 'C', 'F', 'H', 'N', 'O', 'S', 'Cl', 'P', 'Si'] ):
    # average output
    # https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html
    """
    Generate the average SOAP, i.e. the average of the SOAP vectors is
    a global of the molecule.
    """
    average_soap = SOAP(
        r_cut=6.0,
        n_max=8,
        l_max=6,
        average="inner",
        species=species,
        sparse=False,
    )

    molecule = Atoms(numbers=chgs, positions=crds)
    return average_soap.create(molecule)


def get_all_spham(CHARGES,COORDINATES, pad=400):
    X = []
    xyz_file_path = './.tmp.xyz' 
    for z,r in zip(CHARGES,COORDINATES):
        # Create a temporary file to write the XYZ data
        with open(xyz_file_path, "w") as file:
            file.write(f"{z} {r[0]} {r[1]} {r[2]}\n")
        pdb.set_trace()
        # Load the molecule from the XYZ file
        mol = compound.xyz_to_mol(xyz_file_path, "def2svp", charge=0, spin=0)

        X.append(spahm.compute_spahm.get_spahm_representation(mol, "lb")[0])

    X = np.array(X)
    X = pad_max_size(X, pad)

    return X


def get_cMBDF(CHARGES, COORDINATES, local=False):
    X = generate_mbdf(CHARGES, COORDINATES, local=local)
    return X


"""
if __name__ == '__main__':


    X = generate_mbdf(z, R, local=False)

    converter = ZipFeaturizer()
    string_reps = converter.bin_vectors(X)
    pdb.set_trace()

    
"""
