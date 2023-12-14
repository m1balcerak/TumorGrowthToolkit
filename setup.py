from setuptools import setup, find_packages

setup(
    name='forwardFK_FDM',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy'
    ],
    author='Michal Balcerak',
    author_email='m1balcerak@gmail.com',
    description='FK FDM solver - simulating tumor growth in the brain.'
)
