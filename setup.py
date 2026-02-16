from setuptools import setup, find_packages

setup(
    name='toneforge',
    version='0.1.0',
    packages=find_packages(include=['toneforge', 'toneforge.*']),
    install_requires=[
        'numpy',
        'colour-science',
    ],
    author='albert',
    description='Local HDRI processing library with color space, tone mapping, and gamma tools',
)
