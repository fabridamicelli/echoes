from setuptools import setup, find_packages


DESCRIPTION = 'Machine Learning with Echo State Networks in Python'
with open('README.md', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

base_packages = [
    "numpy>=1.16.0",
    "scikit-learn>=0.20.2",
    "pandas>=1.0.3",
    "matplotlib>=3.2.0",
    "seaborn>=0.10.1",
    "numba==0.54.1"
]

test_packages = [
    "pytest>=4.0.2",
    "mypy>=0.770",
]


setup(
    name='echoes',
    version='0.0.6',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Fabrizio Damicelli',
    author_email='fabridamicelli@gmail.com',
    url="https://github.com/fabridamicelli/echoes",
    packages=find_packages(exclude=['notebooks', 'docs']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.6',
    install_requires=base_packages,
    include_package_data=True
)
