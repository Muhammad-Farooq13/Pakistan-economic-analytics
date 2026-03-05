"""setup.py – makes the src/ package importable project-wide."""

from setuptools import setup, find_packages

setup(
    name="pakistan-economic-analytics",
    version="1.0.0",
    description="GDP Growth Forecasting & Economic Dashboard for Pakistan (2000-2025)",
    author="Muhammad Farooq",
    author_email="mfarooqshafee333@gmail.com",
    url="https://github.com/Muhammad-Farooq-13/pakistan-economic-analytics",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.1",
        "scikit-learn>=1.4",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "xgboost>=2.0",
        "shap>=0.44",
        "streamlit>=1.32",
        "joblib>=1.3",
        "statsmodels>=0.14",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
