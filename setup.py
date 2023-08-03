import sys

from setuptools import setup, find_packages


# Check python version
MINIMAL_PY_VERSION = (3, 8)
if sys.version_info < MINIMAL_PY_VERSION:
    raise RuntimeError('This app works only with Python {}+'.format('.'.join(map(str, MINIMAL_PY_VERSION))))

setup(
    name='sats_receiver',
    setuptools_git_versioning={
        'enabled': True,
        'template': '{tag}.{ccount}',
        'dev_template': '{tag}.{ccount}',
        'dirty_template': '{tag}.{ccount}',
    },
    url='https://github.com/baskiton/sats-receiver',
    project_urls={
        'Bug Tracker': "https://github.com/baskiton/sats-receiver/issues",
    },
    license='MIT',
    author='Alexander Baskikh',
    python_requires='>=3.7',
    author_email='baskiton@gmail.com',
    description='Satellites data receiver based on GNU Radio',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={
        '': ['resources/*'],
    },
    install_requires=[
        'ephem',
        'numpy',
        'Pillow',
        'pyshp',
        'python-dateutil',
        'scipy',
    ],
    setup_requires=[
        'setuptools-git-versioning',
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Environment :: Console',
        'Programming Language :: Python :: 3',
        'Topic :: Communications :: Ham Radio',
    ]
)
