from setuptools import setup, find_packages

setup(
    name='rigCal',            # package name
    version='0.1',
    packages=find_packages(where="src"),  # find packages inside src
    package_dir={"": "src"},     # tells Python that packages are under src
)