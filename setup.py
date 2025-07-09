from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    with open(file_path, "r") as file_obj: 
        requirements = file_obj.readlines()
        requirements = [i.strip() for i in requirements if i.strip() != "-e ."]
    return requirements

setup(
    name='pneufusion',  
    version='0.0.1',
    author='Prabhjot Singh',
    author_email='prabh2004p@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt') 
)
