#!/usr/bin/env python3
import os
os.environ.pop("ALL_PROXY", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

from vllm_agent_eval.cli import main

if __name__ == "__main__":
    main()
