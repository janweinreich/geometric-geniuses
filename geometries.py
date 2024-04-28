#http://www.sgdml.org/#code
#https://github.com/dkhan42/MBDF/tree/main
import random
random.seed(666)
import numpy as np
import os
from cMBDF import generate_mbdf
import pdb


#http://www.quantum-machine.org/gdml/data/npz/md17_aspirin.npz
#if file is not found, download it from the above link and place it in the same directory as the code
if not os.path.exists('md17_aspirin.npz'):
    print('Downloading the file')
    import urllib.request
    url = 'http://www.quantum-machine.org/gdml/data/npz/md17_aspirin.npz'
    urllib.request.urlretrieve(url, 'md17_aspirin.npz')
    print('Download complete')
else:
    print('File already exists')

data = np.load('md17_aspirin.npz')

print('Data keys:', data.keys())


R = data['R']
z = (np.array(list(data['z'])*R.shape[0])).reshape(R.shape[0], -1)
E = data['E']
F = data['F']

indices = np.random.choice(R.shape[0], 1000, replace = False)

z = z[indices]
R = R[indices]

X = generate_mbdf(z, R, local=False)

pdb.set_trace()