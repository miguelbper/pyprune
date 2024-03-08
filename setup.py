from setuptools import setup, find_packages

setup(
    name='backtracking',
    version='0.1.0',
    url='https://github.com/miguelbper/backtracking',
    author='Miguel Pereira',
    author_email='miguel.b.per@gmail.com',
    description='Backtracking algorithm for constraint satisfaction puzzles.',
    packages=find_packages(),
    install_requires=['numpy', 'numba'],
)
