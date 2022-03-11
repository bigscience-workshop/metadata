from setuptools import find_packages, setup


def req_file(filename):
    with open(filename) as f:
        content = f.readlines()
    return [x.strip() for x in content]


install_requires = req_file("requirements.txt")

preprocessing_requires = {
    "html": ["lxml==4.6.3", "htmlmin==0.1.12"],
    "entity": ["REL @ git+https://github.com/manandey/REL.git#egg=REL"],
    "timestamp": ["bs_dateutil @ git+https://github.com/cccntu/dateutil.git@2.8.5"],
    "website_description": ["wikipedia2vec==1.0.5", "nltk==3.6.7"],
}

preprocessing_dependencies = []
for dependencies in preprocessing_requires.values():
    preprocessing_dependencies.extend(dependencies)

setup(
    name="bsmetadata",
    python_requires=">=3.7.11, <3.9",  # wikipedia2vec doesn't support Python 3.9
    version="0.1.0",
    url="https://github.com/bigscience-workshop/metadata.git",
    author="Multiple Authors",
    author_email="xxx",
    description="Codebase for including metadata (e.g., URLs, timestamps, HTML tags) during language model pretraining.",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={"preprocessing": preprocessing_dependencies, "torch": "torch==1.9.0"},  # for Flair
)
