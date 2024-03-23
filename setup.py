from setuptools import setup, find_packages
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='pyprune',
    version='0.1.0',
    url='https://github.com/miguelbper/pyprune',
    author='Miguel Pereira',
    author_email='miguel.b.per@gmail.com',
    description='Backtracking algorithm for constraint satisfaction puzzles.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=['numpy', 'numba'],
    project_urls={
        'Source': 'https://github.com/miguelbper/pyprune',
    },
)
