#!/usr/bin/env python
"""Fetch the pinned tau2-bench data directory. Same as `uv run tooluse-rl-fetch-data`."""

import sys

from tooluse_rl.data import main

if __name__ == "__main__":
    sys.exit(main())
