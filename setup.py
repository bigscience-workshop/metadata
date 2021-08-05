from setuptools import find_packages, setup


def req_file(filename):
    with open(filename) as f:
        content = f.readlines()
    return [x.strip() for x in content]


install_requires = req_file("requirements.txt")

setup(
    name="bsmetadata",
    version="0.0.0",
    url="https://github.com/bigscience-workshop/metadata.git",
    author="Multiple Authors",
    author_email="xxx",
    description="Codebase for including metadata (e.g., URLs, timestamps, HTML tags) during language model pretraining.",
    packages=find_packages(),
    install_requires=install_requires,
)
