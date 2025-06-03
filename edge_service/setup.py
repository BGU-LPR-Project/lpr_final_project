from setuptools import setup, find_packages

setup(
    name="edge_service",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy",
        "ultralytics",
        "redis",
    ],
)
