from setuptools import setup, find_packages

setup(
    name='mechanics',
    version="0.0.1",
    description="a library to model and calculate mechanics",
    author='Yuki Fujihara',
    packages=find_packages(),
    license='MIT',
    install_requires=[
        'numpy',
        'sympy',
        'matplotlib',
        'shortuuid'
    ],
    package_data={
        'mechanics': ['mechanics/fortran/*.f90'],
    },
)