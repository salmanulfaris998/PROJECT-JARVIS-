#!/usr/bin/env python3
"""
JARVIS AI - Nothing Phone 2a Root-Level Integration
Setup script for installation and distribution
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Project metadata
PROJECT_NAME = "jarvis-ai-nothing-phone"
PROJECT_VERSION = "2.0.0"
PROJECT_DESCRIPTION = "Transform your Nothing Phone 2a into a real JARVIS AI with root-level device control"
PROJECT_LONG_DESCRIPTION = read_readme()
PROJECT_LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
PROJECT_AUTHOR = "JARVIS AI Development Team"
PROJECT_AUTHOR_EMAIL = "jarvis.ai@nothing.tech"
PROJECT_URL = "https://github.com/yourusername/jarvis-ai-nothing-phone"
PROJECT_DOWNLOAD_URL = "https://github.com/yourusername/jarvis-ai-nothing-phone/archive/main.zip"
PROJECT_CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: End Users/Desktop",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: System :: Hardware",
    "Topic :: System :: Operating System",
    "Topic :: System :: Systems Administration",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Operating System :: Android",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows",
    "Framework :: Pytest",
    "Topic :: Multimedia :: Sound/Audio :: Speech",
    "Topic :: Communications :: Voice",
    "Topic :: Home Automation",
    "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: System :: Archiving :: Backup",
    "Topic :: System :: Monitoring",
    "Topic :: System :: Networking :: Monitoring",
    "Topic :: System :: Recovery",
    "Topic :: System :: Systems Administration :: Monitoring",
    "Topic :: Utilities",
]

# Package configuration
PACKAGES = find_packages(include=[
    "phase1_foundation*",
    "phase2_hybrid_ultimate*", 
    "phase3_system_integration*",
    "phase4_ai_brain*",
    "phase5_phone_transformation*",
    "phase6_advanced_features*",
    "phase7_optimization*",
    "tools*"
])

# Entry points
ENTRY_POINTS = {
    "console_scripts": [
        "jarvis=phase1_foundation.advanced_ai_brain:main",
        "jarvis-voice=phase1_foundation.neural_voice_system:main",
        "jarvis-control=phase2_hybrid_ultimate.proven_foundation.device_control_mastery:main",
        "jarvis-setup=phase1_foundation.m2_setup:main",
    ],
}

# Setup configuration
setup(
    name=PROJECT_NAME,
    version=PROJECT_VERSION,
    author=PROJECT_AUTHOR,
    author_email=PROJECT_AUTHOR_EMAIL,
    description=PROJECT_DESCRIPTION,
    long_description=PROJECT_LONG_DESCRIPTION,
    long_description_content_type=PROJECT_LONG_DESCRIPTION_CONTENT_TYPE,
    url=PROJECT_URL,
    download_url=PROJECT_DOWNLOAD_URL,
    packages=PACKAGES,
    classifiers=PROJECT_CLASSIFIERS,
    python_requires=">=3.12",
    install_requires=read_requirements(),
    entry_points=ENTRY_POINTS,
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.ini", "*.cfg", "*.log", "*.db"],
    },
    zip_safe=False,
    keywords=[
        "jarvis", "ai", "artificial-intelligence", "nothing-phone", 
        "android", "root", "voice-assistant", "device-control",
        "automation", "machine-learning", "neural-networks",
        "voice-recognition", "system-integration", "hardware-control"
    ],
    project_urls={
        "Bug Reports": f"{PROJECT_URL}/issues",
        "Source": PROJECT_URL,
        "Documentation": f"{PROJECT_URL}/wiki",
        "Changelog": f"{PROJECT_URL}/blob/main/CHANGELOG.md",
        "Contributing": f"{PROJECT_URL}/blob/main/CONTRIBUTING.md",
    },
    # Additional metadata
    maintainer=PROJECT_AUTHOR,
    maintainer_email=PROJECT_AUTHOR_EMAIL,
    license="MIT",
    platforms=["Android", "Linux", "macOS", "Windows"],
    requires_python=">=3.12",
    # Development dependencies
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "pre-commit>=3.3.0",
        ],
        "test": [
            "pytest>=7.3.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.10.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "mkdocs>=1.4.0",
            "mkdocs-material>=9.1.0",
        ],
    },
)

if __name__ == "__main__":
    print(f"Setting up {PROJECT_NAME} v{PROJECT_VERSION}")
    print(f"Description: {PROJECT_DESCRIPTION}")
    print(f"Packages found: {len(PACKAGES)}")
    print("Setup complete!")
