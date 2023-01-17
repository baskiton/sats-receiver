import pathlib
import sys

from setuptools import setup, find_packages


# Check python version
MINIMAL_PY_VERSION = (3, 7)
if sys.version_info < MINIMAL_PY_VERSION:
    raise RuntimeError('This app works only with Python {}+'.format('.'.join(map(str, MINIMAL_PY_VERSION))))

HERE = pathlib.Path(__file__).parent


def get_version():
    for line in (HERE / 'sats_receiver' / '__init__.py').open():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]


def get_long_description():
    return (HERE / 'README.md').read_text('utf-8')


setup(
    name='sats_receiver',
    version=get_version(),
    url='https://github.com/baskiton/sats-receiver',
    project_urls={
        'Bug Tracker': "https://github.com/baskiton/sats-receiver/issues",
    },
    license='MIT',
    author='Alexander Baskikh',
    python_requires='>=3.7',
    author_email='baskiton@gmail.com',
    description='Satellites data receiver based on GNU Radio',
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'ephem',
        'numpy',
        'Pillow',
        'pyshp',
        'python-dateutil',
        'scipy',
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Environment :: Console',
        'Programming Language :: Python :: 3',
        'Topic :: Communications :: Ham Radio',
    ]
)
