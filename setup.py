from setuptools import setup, find_packages

setup(
    name='chroptiks',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'matplotlib', 'numpy', 'scipy'  # Add your dependencies here
    ],
    # Additional metadata
    author='Chris Agostino',
    author_email='cjp.agostino@gmail.com',
    description='A set of helper functions for quickly making matplotlib plots using latex fonts andwith a high number of built-in options.',
    license='MIT',
    keywords='matplotlib plotting utilities',
    url='https://github.com/cagostino/chroptiks', # Optional
)
