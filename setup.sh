#!/bin/sh
git submodule update --init --recursive
uv sync
uv pip install --no-build-isolation -e ./thirdparty/executorch --verbose
uv pip install --no-build-isolation -e ./thirdparty/optimum-executorch
