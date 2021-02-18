from setuptools import setup, find_packages
import re

VERSIONFILE = "Neurosetta/__init__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
	verstr=mo.group(1)
else:
	raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open('requirements.txt') as f:
	requirements = f.read().splitlines()
	requirements = [l for l in requirements if not l.startswith('#')]

setup(name='Neurosetta',
	version=verstr,
	description='Converting between different neuron morphology data structures',
	url='https://github.com/NikDrummond/Neurosetta',
	author='Nik Drummond',
	author_email='nikolasdrummond@gmail.com',
	license='MIT',
	packages=find_packages(),
	zip_safe=False
)
