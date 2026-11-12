from setuptools import setup, find_packages

setup(
    name="stock-analysis-local",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "peft",
        "bitsandbytes",
        "trl",
        "accelerate",
        "datasets",
        "pandas",
        "numpy",
        "yfinance",
        "ta",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "rich",
        "python-dotenv",
        "sentence-transformers",
        "faiss-cpu",
    ],
)
