from setuptools import find_packages, setup

with open("core/README.md", "r") as f:
    long_description = f.read()

setup(
    name="optivarproxy",
    version="0.0.1",
    description="Library for optimal sub-Gaussian proxy variance computations.",
    package_dir={"": "core"},
    packages=find_packages(where="core"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="gihub_link_to_be_inserted",
    author="Soufiane atouani",
    author_email="soufiiane.atouanii@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy==1.23.5"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.10",
)

