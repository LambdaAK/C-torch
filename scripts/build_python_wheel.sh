#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cmake -B "${ROOT_DIR}/${BUILD_DIR}" -DCTORCH_BUILD_TESTS=OFF -DCTORCH_BUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release "${ROOT_DIR}"
cmake --build "${ROOT_DIR}/${BUILD_DIR}" --target ctorch_c --parallel

find_ctorch_lib() {
    local candidates=(
        "${ROOT_DIR}/${BUILD_DIR}/libctorch_c.dylib"
        "${ROOT_DIR}/${BUILD_DIR}/libctorch_c.so"
        "${ROOT_DIR}/${BUILD_DIR}/ctorch_c.dll"
        "${ROOT_DIR}/${BUILD_DIR}/libctorch_c.dll"
        "${ROOT_DIR}/${BUILD_DIR}/Release/ctorch_c.dll"
        "${ROOT_DIR}/${BUILD_DIR}/Release/libctorch_c.dll"
        "${ROOT_DIR}/${BUILD_DIR}/Debug/ctorch_c.dll"
        "${ROOT_DIR}/${BUILD_DIR}/Debug/libctorch_c.dll"
    )
    local p=""
    for p in "${candidates[@]}"; do
        if [[ -f "${p}" ]]; then
            printf '%s\n' "${p}"
            return 0
        fi
    done
    return 1
}

CTORCH_LIB_PATH="$(find_ctorch_lib)"
if [[ -z "${CTORCH_LIB_PATH}" ]]; then
    echo "Could not locate built ctorch_c shared library under ${ROOT_DIR}/${BUILD_DIR}" >&2
    exit 1
fi

cp -f "${CTORCH_LIB_PATH}" "${ROOT_DIR}/python/ctorch/$(basename "${CTORCH_LIB_PATH}")"

mkdir -p "${ROOT_DIR}/dist"
"${PYTHON_BIN}" -m pip wheel "${ROOT_DIR}" --no-deps --no-build-isolation -w "${ROOT_DIR}/dist"

echo "Wheel build complete. Artifacts in ${ROOT_DIR}/dist"
