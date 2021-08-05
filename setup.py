from setuptools import find_packages, setup


def req_file(filename):
    with open(filename) as f:
        content = f.readlines()
    return [x.strip() for x in content]


install_requires = req_file("requirements.txt")

# TODO Fill
setup(
    name="MyPackageName",
    version="1.0.0",
    url="https://github.com/bigscience-workshop/metadata.git",
    author="Author Name",
    author_email="author@gmail.com",
    description="Description of my package",
    packages=find_packages(),
    install_requires=install_requires,
)
