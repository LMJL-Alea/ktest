import sys
import platform
from setuptools import setup

# Check for unsupported macOS x86 + Python >= 3.13
if (
    sys.platform == "darwin" and platform.machine() == "x86_64" and
    sys.version_info >= (3, 13)
):
    raise SystemExit(
        "❌ Installation failed: This package does not support macOS x86 with Python >= 3.13. Please use Python < 3.13 or switch to a supported platform (Linux/macOS ARM)."
    )

setup()
