from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="alarm",
    license="Apache 2.0",
    version="0.1.0",
    description="An alignment framework for by hierarchical rewards modeling",
    author="Yuhang Lai",
    packages=find_packages(),
    keywords="ppo, transformers, alignment, language modeling, rlhf",
    python_requires=">=3.9",
    install_requires=requirements,
    url="https://github.com/halfrot/ALaRM",
    author_email="laiyuhang01@gmail.com"
)
