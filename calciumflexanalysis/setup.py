import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="plate-map-LJC-SW", # Replace with your own username
    version="0.0.1",
    author="Stuart Warriner, Lawrence Collins",
    author_email="s.l.warriner@leeds.ac.uk, lawrencejordancollins@gmail.com",
    description="Plate map uploading, processing & visualisaion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)