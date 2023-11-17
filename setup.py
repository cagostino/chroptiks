from setuptools import setup, find_packages

setup(
    name='chroptiks',
    version='0.1.2',
    packages=find_packages(),
    install_requires=[
        'matplotlib', 'numpy', 'scipy'  # Add your dependencies here
    ],
    # Additional metadata
    author='Chris Agostino',
    author_email='cjp.agostino@gmail.com',
    long_description=open('README.md').read(),  # or README.rst if using RST
    long_description_content_type='text/markdown',

    description='A set of helper functions for quickly making matplotlib plots using latex fonts and with a high number of built-in options.',
    license='MIT',
    keywords='matplotlib plotting utilities',
    url='https://github.com/cagostino/chroptiks', # Optional
)
