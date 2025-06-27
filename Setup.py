from setuptools import setup, find_packages

setup(
    name="pegasus-arxiv-summarizer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Fine-tuned Pegasus model for ArXiv paper summarization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pegasus-arxiv-summarizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "evaluate>=0.4.0",
    ],
)
