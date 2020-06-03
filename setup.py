from setuptools import setup, find_packages


setup(
    name='echoes',
    version='0.0.1',
    description='Machine Learning with Echo State Networks',
    author='Fabrizio Damicelli',
    author_email='f.damicelli@uke.de',
    url="https://github.com/fabridamicelli/echoes",
    packages=find_packages(exclude=['notebooks', 'docs']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.6',
)
