from setuptools import setup, find_packages

setup(
    name="rlgames",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    author="Arya Gopikrishnan",
    description="A collection of reinforcement learning algorithms and environments",
    python_requires=">=3.6",
) 