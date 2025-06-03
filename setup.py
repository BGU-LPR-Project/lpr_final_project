from setuptools import setup, find_packages

setup(
    name="lpr",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "ultralytics",
        "redis",
        "fastapi",
        "uvicorn",
        "requests",
        "scipy"
    ]
)
