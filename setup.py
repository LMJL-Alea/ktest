import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ktest-AnthoOzier", # Replace with your own username
    version="0.0.1",
    author="Anthony Ozier-Lafontaine",
    author_email="anthony.ozier-lafontaine@ec-nantes.fr",
    description="Package implementing efficient kernel tests such as MMD or mKFDR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AnthoOzier/ktest",
    project_urls={
        "Bug Tracker": "https://github.com/AnthoOzier/ktest/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)


