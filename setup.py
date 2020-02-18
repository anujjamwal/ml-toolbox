from setuptools import setup, find_packages

setup(
    name='ml-toolbox',
    version='0.1',
    url='https://github.com/anujjamwal/ml-toolbox.git',
    license='MIT',
    author='Anuj Jamwal',
    author_email='anuj.jamwal1@gmail.com',
    description='Convenience functions to help with machine learning',
    packages=find_packages(exclude=['tests']),
    long_description=open('README.md').read(),
    zip_safe=False
)
