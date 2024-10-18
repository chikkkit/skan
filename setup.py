from setuptools import setup, find_packages

setup(
    name="skan",
    version="0.1.0",
    author="Zhijie Chan",
    author_email="zhijiechencs@gmail.com",
    description="Python libaray for SKAN (Single-parameterized KAN)",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/chikkkit/SKAN",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # 'numpy',
        # 'requests',
    ],
)