#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

NOTEBOOKS_DIR=$1
NOTEBOOKS=$(find "${NOTEBOOKS_DIR}" -name "*.ipynb" -print)
for notebook in $NOTEBOOKS; do
    echo "Running $notebook"
    uv run jupyter nbconvert --execute "$notebook" --inplace
    error_code=$?
    if [ "${error_code}" -ne 0 ]; then
        echo "Error running $notebook"
        exit 1
    fi
done
