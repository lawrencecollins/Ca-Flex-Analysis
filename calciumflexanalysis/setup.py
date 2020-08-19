import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="calciumflexanalysis", 
    version="0.0.3",
    author="Lawrence Collins, Stuart Warriner",
    author_email="s.l.warriner@leeds.ac.uk, lawrencejordancollins@gmail.com",
    description="Processing and analysis of calcium flex assays",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lawrencecollins/Ca-Flex-Analysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    setup_requires=['wheel'],
    install_requires=['platemapping']
)