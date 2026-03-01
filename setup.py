from setuptools import setup, find_packages

setup(
    name='foofit',
    version='0.2.5',
    description='X-ray reflectivity fitting using Parratt formalism',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'lmfit',
        'prettytable',
        'tqdm',
        'corner',
        'joblib',
        'astropy',
        'numdifftools',
    ],
    package_data={'foofit': ['*.xrr']},
)
