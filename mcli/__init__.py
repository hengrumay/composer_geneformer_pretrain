"""Minimal stub for `mcli` to satisfy Composer imports on Databricks Serverless GPU.

Composer's MosaicML logger imports `mcli` at import time, even if you never use the
MosaicMLLogger. On Serverless GPU, installing `mosaicml-cli` can cause dependency
conflicts (notably around `prompt-toolkit` / `questionary` vs runtime `ipython`).

This stub is intentionally tiny: it only aims to make `import mcli` succeed.
If you actually want to use MosaicMLLogger, remove this stub and install `mosaicml-cli`.
"""

__all__ = [
    "__version__",
]

__version__ = "0.0.0-stub"

