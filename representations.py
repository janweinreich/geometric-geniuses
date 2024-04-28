import numpy as np
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid, KFold
from qml.representations import generate_fchl_acsf
from MBDF import generate_mbdf, generate_df
from dscribe.descriptors import SOAP
from ase import Atoms
from joblib import Parallel, delayed
import glob
from qml.representations import get_slatm_mbtypes, generate_slatm
import bz2, pickle
import json
from qstack import compound, spahm
from qml.representations import generate_bob


#lookup here!
#https://github.com/janweinreich/FML
#https://github.com/janweinreich/FML/blob/master/activation/activation_oneout_cv_krr_mbdf_gzip.py

def gen_all_bob(COORDINATES, NUC_CHARGES, size=158, asize = {"O":3, "C":7, "N":3, "H":16, "S":1}):
    X = []
    for i in tqdm(range(len(COORDINATES))):
        X.append(generate_bob(NUC_CHARGES[i], COORDINATES[i],atomtypes=[el for el in asize.keys()],  size=size, asize = asize))
    
    X= np.array(X)
    
    return X



def numpy_encoder(obj):
    """ Special JSON encoder for numpy types """
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, 
        np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return json.JSONEncoder().default(obj)


def save_dict_to_json(d, filename):
    with open(filename, 'w') as f:
        json.dump(d, f, default=numpy_encoder, indent=4)


def load_json_to_dict(filename, convert_to_numpy=False):
    with open(filename, 'r') as f:
        d = json.load(f)
        
    if convert_to_numpy:
        def convert(item):
            if isinstance(item, list):
                return np.array(item)
            if isinstance(item, dict):
                return {key: convert(value) for key, value in item.items()}
            return item
        
        d = convert(d)
        
    return d


def farthest_point_sampling(points, N, initial_point=None):
    farthest_points = np.zeros((N, 3))
    farthest_point_indices = np.zeros(N, dtype=int)
    
    if initial_point is not None:
        farthest_points[0] = initial_point
        farthest_point_indices[0] = np.where((points==initial_point).all(axis=1))[0][0]
    else:
        random_index = np.random.choice(len(points))
        farthest_points[0] = points[random_index]
        farthest_point_indices[0] = random_index
    
    distances = np.linalg.norm(points - farthest_points[0], axis=1)
    
    for i in range(1, N):
        farthest_point_index = np.argmax(distances)
        farthest_points[i] = points[farthest_point_index]
        farthest_point_indices[i] = farthest_point_index
        distances = np.minimum(distances, np.linalg.norm(points - farthest_points[i], axis=1))
    
    return farthest_points, farthest_point_indices

def energy_point_sampling(points, N, energies):
    """
    Sample N points from points with probability proportional to their energies
    """
    energies = np.array(energies)
    
    #order with ascending energies
    order = np.argsort(energies)[:N]
    
    points = points[order]
    energies = energies[order]

    return points,order





compress_fileopener = {True: bz2.BZ2File, False: open}
pkl_compress_ending = {True: ".pkl.bz2", False: ".pkl"}


def dump2pkl(obj, filename: str, compress: bool = False):
    """
    Dump an object to a pickle file.
    obj : object to be saved
    filename : name of the output file
    compress : whether bz2 library is used for compressing the file.
    """
    output_file = compress_fileopener[compress](filename, "wb")
    pickle.dump(obj, output_file)
    output_file.close()


def loadpkl(filename: str, compress: bool = False):
    """
    Load an object from a pickle file.
    filename : name of the imported file
    compress : whether bz2 compression was used in creating the loaded file.
    """
    input_file = compress_fileopener[compress](filename, "rb")
    obj = pickle.load(input_file)
    input_file.close()
    return obj


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


def train(representations,Q, y, kernel_sigma, kernel_lambda):
    from qml.kernels import get_local_symmetric_kernel #get_local_kernel
    Ktrain = get_local_symmetric_kernel(representations, Q, kernel_sigma) 
    for i in range(Ktrain.shape[0]):
        Ktrain[i, i] -= kernel_lambda

    alphas = np.linalg.solve(Ktrain, y)
    return alphas


def predict(representations, representations_test,Q1, QQ, alphas, kernel_sigma):
    from qml.kernels import get_local_kernel

    K_geo = get_local_kernel(
        representations, representations_test, Q1, QQ, kernel_sigma)

    predvals = np.dot(K_geo, alphas)
    
    return predvals

def mae(prediction, reference):
    return np.mean(np.abs(prediction - reference))


def CV(representations, Q, y, param_grid, kfn=5, refit=True):
    
    """
    Perform as Cross Validation to obtain the optimal weight coefficients for
    each trainingset sizes. A grid search will be performed.

    input:
    representation, distances for training
    array of hyperparameters

    returns:
    weight coefficients and optimized hyperparameters
    """
    
    param_grid = list(ParameterGrid(param_grid))

    kf = KFold(n_splits=kfn, shuffle=True, random_state=42)

    all_gpr = []
    for gp in tqdm(param_grid):
        gp_errs = []
        for train_index, test_index in kf.split(representations):
            alphas = train(representations[train_index],Q[train_index], y[train_index],kernel_sigma=gp['kernel_sigma'], kernel_lambda=gp['kernel_lambda'])

            predictions = predict(representations[train_index], representations[test_index],Q[train_index], Q[test_index],
                                  alphas, kernel_sigma=gp['kernel_sigma'])

            gp_errs.append(
                mae(predictions, y[test_index]))

        all_gpr.append(np.mean(np.array(gp_errs)))

    all_gpr = np.array(all_gpr)
    opt_p = param_grid[np.argmin(all_gpr)]
    print(opt_p)

    if refit:
        alphas_opt = train(representations,Q, y,
                           kernel_sigma=opt_p['kernel_sigma'], kernel_lambda=opt_p['kernel_lambda'])
        return alphas_opt, opt_p



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


def gen_mbdf(crds, chgs, binsize = 0.8):
    mbdf =  generate_mbdf(np.array([chgs]), np.array([crds]))
    
    return mbdf, generate_df(mbdf,np.array([chgs]), binsize = 0.8)

def check_xyz_files(path, filenames, y_values):
    
    NAMES, CONFORMERS, Y = [], [], []
    NAME_FAIL = []
    for name, yval in zip(filenames,y_values):
        try:
        
            subfolder    = glob.glob(f"{path}conformers_{name}*/")
            xyz_filename = glob.glob(f"{subfolder[0]}/*conformers*")[0]
            CONFORMERS.append(read_xyz(xyz_filename, get_energy=True))
            NAMES.append(name)
            Y.append(yval)

        except Exception as e:
            print(e)
            NAME_FAIL.append(name)

    Y = np.array(Y).reshape(-1,1)
    return NAMES, CONFORMERS, Y, NAME_FAIL


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




def read_xyz(filepath, get_energy = False):
    conformers, energies = [],[]

    with open(filepath, 'r') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        try:
            # get the number of atoms
            num_atoms = int(lines[i].strip())
            i += 1

            # get the energy
            if get_energy:
                energy = float(re.search(r"Energy = ([-+]?\d*\.\d+|\d+) eV", lines[i]).group(1))
                energies.append(energy)
                
                #float(lines[i].strip())
            else:
                energy = None
            i += 1

            # get the atom symbols and coordinates
            elements, coordinates = [], []
            for _ in range(num_atoms):
                line = lines[i].strip().split()
                atom = line[0]
                elements.append(atom)   
                coordinates.append(list(map(float, line[1:])))
                i += 1

            # add the conformer to the list
            conformers.append({"elements": elements, "coordinates": coordinates})

        except (IndexError, ValueError) as e:
            print(f"Error reading file {filepath} at line {i+1}: {e}")
            break

    
    if get_energy:
        return conformers, np.array(energies)
    else:
        return conformers



def pad_max_size(X, max_size):
    X= np.array([np.pad(subarray, (0, max_size - len(subarray)), 'constant') for subarray in X])
    return X

def get_all_spham(MOLS, pad=400):
    X = []
    for mol in MOLS:
        X.append(spahm.compute_spahm.get_spahm_representation(mol, "lb")[0])

    X = np.array(X)
    X = pad_max_size(X, pad)

    return X