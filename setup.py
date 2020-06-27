from setuptools import setup, find_packages


with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    # Package Info
    name='revgraph',
    author='shhoalex',
    version='0.0.1',
    url='https://github.com/shhoalex/revgraph',
    description='A toy deep learning library built using numpy as its only dependency.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],

    # Dependencies
    install_requires=['numpy>=1.18.2', 'dill>=0.3.2'],
    python_requires='>=3.6',

    # Exporting the following packages
    py_modules=find_packages(where='revgraph'),
    package_dir={'': 'revgraph'},
    package_data={
        # Configs for the package 'revgraph.dataset'
        '': ['*.data', '*.gz']
    },
)
