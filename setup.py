from setuptools import setup, find_packages

setup(
    name="duet-vlm",
    version="1.0.0",
    description="DUET-VLM: Unified Vision-Language Model Efficiency (VisionZip + PyramidDrop)",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="DUET-VLM Team",
    author_email="yangsenqiao.ai@gmail.com",
    url="https://github.com/dvlab-research/duet-vlm",
    python_requires=">=3.8",
    packages=find_packages(include=[
        "llava", "llava.*",
        "visionzip", "visionzip.*",
        "videollava", "videollava.*",
        "qwen2_5_vl", "qwen2_5_vl.*",
    ]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # Core dependencies
        "torch>=2.0",
        "torchvision",
        "transformers>=4.37",
        "accelerate>=0.28.0",
        "peft>=0.10.0",
        
        # Tokenization and NLP
        "sentencepiece",
        "tiktoken",
        "spacy",
        "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
        "nltk",
        "shortuuid",
        
        # Image processing
        "pillow",
        
        # Utilities
        "tqdm",
        "requests",
        "einops",
        "mpi4py==4.1.1",
        "openai==0.28.0",
        
        # Optional but commonly used
        "wandb",
        "openpyxl",
        "huggingface_hub[hf_xet]",
        "ipdb",
    ],
    extras_require={
        # Video-LLaVA specific dependencies
        "video": [
            "decord",          # Video loading
            "einops",          # Tensor operations for LanguageBind
            "gradio",          # Web UI
        ],
        # Qwen2.5-VL specific dependencies
        "qwen": [
            "qwen-vl-utils",   # Qwen VL utilities
        ],
        # Full installation with all features
        "all": [
            "decord",
            "einops",
            "gradio",
            "qwen-vl-utils",
            "openai",
            "mpi4py",
            "deepspeed",
        ],
        # Development dependencies
        "dev": [
            "pytest",
            "black",
            "isort",
            "flake8",
        ],
    },
)
