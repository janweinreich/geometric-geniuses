from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="Feat2LLM",
    version="0.1.0",
    author="Jan Weinreich, Ankur Gupta, Amir Naghdi, Alishba Imran",
    author_email="jan.weinreich@epfl.ch",
    description="Feat2LLM can take any feature vector and convert it into a string representation. This string representation can then be used as input for a language model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/janweinreich/geometric-geniuses",
    packages=find_packages(include=['Feat2LLM', 'Feat2LLM.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "ase==3.23.0",
        "deepchem==2.5.0",
        "dscribe==2.1.1",
        "joblib==1.3.2",
        "matplotlib==3.8.3",
        "numba==0.59.1",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "rdkit==2023.9.5",
        "Requests==2.32.3",
        "scikit_learn==1.3.0",
        "scipy==1.13.1",
        "selfies==2.1.1",
        "torch==2.2.1",
        "tqdm==4.65.0",
        "transformers==4.41.2",
    ],
    include_package_data=True,
)
