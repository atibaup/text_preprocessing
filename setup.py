from setuptools import setup

setup(
    name='text_preprocessing',
    version='0.0.1',
    packages=['text_preprocessing'],
    install_requires=['numpy >1.16,<2.0'],
    python_requires='>=3.6',
    test_suite='nose.collector',
    tests_require=['nose'],
)