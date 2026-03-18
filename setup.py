from setuptools import setup, find_packages

setup(
    name="aws-product-classifier",
    version="1.0.0",
    description="Production-grade product image classification system on AWS",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/aws-product-classifier",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=10.0.0",
        "boto3>=1.28.0",
        "sagemaker>=2.170.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
