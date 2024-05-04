#!/usr/bin/env python3

import bz2, pickle
import json
import sys
import numpy as np

compress_fileopener = {True: bz2.BZ2File, False: open}

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

def make_json(X, y, filename=None):
    # Create a list of dictionaries
    y = np.squeeze(y)
    data = [
        {"instruction": "What is the solvation energy of molecule {} in kcal/mol?".format(smiles),
        "input": smiles,
        "output": value}
        for smiles, value in zip(X, y)
        ]
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == '__main__':
    data_smi = loadpkl("data/rep_delaney_smiles_selfies.pkl", compress=True)
    X_train_smi, X_test_smi, y_train, y_test = data_smi["X_train"], data_smi["X_test"], data_smi["y_train"], data_smi["y_test"]
    train_data = make_json(X_train_smi, y_train, filename='train_smi.json')
    test_data = make_json(X_test_smi, y_test, filename='test_smi.json')

