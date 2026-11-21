from setuptools import find_packages, setup

setup(
    name="customer-churn",
    version="0.1.0",
    author="Riadh Ghanmi",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
)
