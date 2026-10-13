#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
TASK_DEV=OFF
TASK_MATERIALIZE=0
TASK_DEPENDENCIES_ONLY=0
for TASK_ARG in "$@"; do
  case "$TASK_ARG" in
    --development) TASK_DEV=ON ;;
    --materialize-evidence) TASK_MATERIALIZE=1 ;;
    --dependencies-only) TASK_DEPENDENCIES_ONLY=1 ;;
    *) echo "Unknown build option: $TASK_ARG" >&2; exit 2 ;;
  esac
done
if [[ "$TASK_DEPENDENCIES_ONLY" == 1 && ( "$TASK_MATERIALIZE" == 1 || "$TASK_DEV" == ON ) ]]; then
  echo '--dependencies-only cannot be combined with native build/evidence options' >&2
  exit 2
fi
mkdir -p third_party
if [[ ! -d third_party/SEAL/.git ]]; then
  git clone --depth 1 --branch v4.1.2 https://github.com/microsoft/SEAL.git third_party/SEAL
fi
if [[ "$(git -C third_party/SEAL rev-parse HEAD)" != 119dc32e135cb89c1062076a69310d4413ebc824 ]]; then
  echo 'Locked SEAL commit mismatch' >&2; exit 1
fi
if [[ ! -d third_party/EzPC/.git ]]; then
  git init third_party/EzPC
  git -C third_party/EzPC remote add origin https://github.com/mpc-msri/EzPC.git
  git -C third_party/EzPC fetch --depth 1 origin f24bf3e022dde3493cf0ffe3ebd92dd73936b734
  git -C third_party/EzPC checkout FETCH_HEAD
fi
if [[ "$(git -C third_party/EzPC rev-parse HEAD)" != f24bf3e022dde3493cf0ffe3ebd92dd73936b734 ]]; then
  echo 'Locked SCI commit mismatch' >&2; exit 1
fi
for TASK_DEP in SEAL EzPC; do
  if [[ -n "$(git -C "third_party/$TASK_DEP" diff --name-only HEAD)" ]]; then
    echo "Unexpected tracked dependency edits: $TASK_DEP" >&2; exit 1
  fi
done
cmake -S third_party/SEAL -B third_party/SEAL/build -DCMAKE_BUILD_TYPE=Release \
  -DSEAL_BUILD_DEPS=OFF -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF -DSEAL_USE_ZSTD=OFF \
  -DSEAL_USE_INTEL_HEXL=OFF -DSEAL_USE_GAUSSIAN_NOISE=OFF -DSEAL_DEFAULT_PRNG=Blake2xb -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=OFF \
  -DSEAL_BUILD_TESTS=OFF -DSEAL_BUILD_EXAMPLES=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build third_party/SEAL/build -j1
if [[ "$TASK_DEPENDENCIES_ONLY" == 1 ]]; then
  exit 0
fi
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release -DPRAG_BUILD_ISOLATED_DEVELOPMENT="$TASK_DEV"
cmake --build native/build -j1

"$TASK_ROOT/native/build/prag_bfv_context" "$TASK_ROOT/artifacts/bfv_context.json"
if [[ "$TASK_MATERIALIZE" == 1 ]]; then
  # This explicit step verifies existing run identities; compiling cannot create
  # execution evidence. Probe and worker builds must already match those runs.
  python3 "$TASK_ROOT/tools/build_batch111_evidence.py"
fi
