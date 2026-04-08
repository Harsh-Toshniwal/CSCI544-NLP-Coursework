from setuptools import setup, find_packages

setup(
    name="paragen",
    version="0.1.0",
    description="ParaGen: Controllable Neural Paraphrase Generation with Semantic Preservation",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "nltk>=3.8.1",
        "sentence-transformers>=2.2.2",
        "rouge-score>=0.1.2",
        "scipy>=1.11.2",
        "tqdm>=4.66.1",
        "pyyaml>=6.0",
    ],
)
