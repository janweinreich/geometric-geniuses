#!/usr/bin/env python3

import bz2, pickle
from vec2str import ZipFeaturizer

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

if __name__ == '__main__':
    data = loadpkl('rep_qm7_rdkit.pkl', compress=True)
    X_train = data['X_train']
    y_train = data['y_train']
    print(X_train[0])
    converter = ZipFeaturizer()
    string_reps = converter.bin_vectors(X_train)
    print(string_reps[0])