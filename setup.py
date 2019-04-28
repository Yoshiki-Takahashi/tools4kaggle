from setuptools import setup, find_namespace_packages
import os
import sys

if sys.version_info < (3,5):
    sys.exit('Python < 3.5 is not supported')

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

install_requires = [
    'setuptools',
    'numpy>=1.9.0',
    'sklearn',
    'pandas',
    'seaborn',
    'matplotlib',
    'category_encoders'
]

setup(
    name='t4k',
    version='0.0.1',
    description='Tool Package for Kaggle',
    long_description=read('README.md'),
    author='Yoshiki Takahashi',
    author_email='yoshiki.takahashi.0326@gmail.com',
    install_requires=install_requires,
    url='',
    license=read('LICENSE'),
    packages=find_namespace_packages()
)
