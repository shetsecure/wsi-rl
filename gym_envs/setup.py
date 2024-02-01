from setuptools import setup, find_packages

setup(
    name="gym_envs",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium"
    ],  # Ensure you have gymnasium or gym in your dependencies
)
