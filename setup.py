from setuptools import find_packages, setup

setup(
    name="continual-learner",
    description="Ready-to-use models under one common API",
    version="0.1",
    packages=find_packages(exclude=["docs", "tests", "scripts"]),
    zip_safe=True,
    python_requires=">=3.6, <3.8",
    include_package_data=True,
    install_requires=[
        "click~=7.1.2",
        "scikit-image~=0.17.2",
        "scikit-learn~=0.23.2",
        "pyclipper~=1.2.0",
        "pyyaml~=5.3.1",
        "tensorboardX~=2.1",
        "numpy~=1.19.2",
        "torch~=1.4.0",
        "torchvision~=0.5.0",
        "jsonschema~=3.2.0",
        "pytest~=6.1.1",
        "pandas~=1.1.3",
        "opencv-python~=4.0.0.21",
        "tqdm~=4.50.2",
        "scipy~=1.5.2",
        "munch~=2.5.0",
        "colorlog~=4.2.1",
        "anyconfig~=0.9.11",
        "tensorboard~=2.6.0",
        "python-decouple~=3.4"
    ],
)
