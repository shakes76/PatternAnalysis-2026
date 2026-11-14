#!/usr/bin/env bash

# Fail fast options for robust error handling
set -Eeuo pipefail

BASE_URL="https://isic-archive.s3.amazonaws.com/challenges/2018"
RANGPUR_SRC="/home/groups/comp3710/ISIC2018"

# Resolve ../dataset relative to this script, so it works from any cwd
DATASET_DIR="$(cd "$(dirname "${0}")/.." && pwd)/dataset"

RANGPUR=false

# Always fetched on Rangpur (everything else is copied from the shared copy)
RANGPUR_ZIPS=(
  ISIC2018_Task1_Test_GroundTruth
)
# Fetched when not on Rangpur
ALL_ZIPS=(
  ISIC2018_Task1-2_Training_Input
  ISIC2018_Task1-2_Test_Input
  ISIC2018_Task1_Training_GroundTruth
  ISIC2018_Task1_Test_GroundTruth
)

if [[ -t 1 ]]; then
  MAGENTA=$'\033[35m'
  BLUE=$'\033[34m'
  RESET=$'\033[0m'
else
  MAGENTA=""
  BLUE=""
  RESET=""
fi

usage() {
  cat << EOF
Usage: $(basename "${0}") [-h] [-r]

Fetch ISIC2018 dataset into ${MAGENTA}${DATASET_DIR}${RESET}

Options:
  -h, --help       Display this help message and exit
  -r, --rangpur    Copy the shared dataset from ${MAGENTA}${RANGPUR_SRC}${RESET},
                   then download only ISIC2018_Task1_Test_GroundTruth
EOF
  exit 1
}

parse_params() {
  while [[ $# -gt 0 ]]; do
    case "${1}" in
      -h | --help)
        usage
        ;;
      -r | --rangpur)
        RANGPUR=true
        shift
        ;;
      *)
        echo "Error: Unknown option '${1}'" >&2
        usage
        ;;
    esac
  done
}

# Download <name>.zip from BASE_URL and unzip it into DATASET_DIR
download_and_unzip() {
  local name="${1}"
  local zip="${DATASET_DIR}/${name}.zip"
  echo "Downloading ${BLUE}${name}${RESET}"
  curl -fL --retry 3 -o "${zip}" "${BASE_URL}/${name}.zip"
  echo "Unzipping ${BLUE}${name}${RESET}"
  unzip -q -o "${zip}" -d "${DATASET_DIR}"
  rm -f "${zip}"
}

parse_params "$@"
mkdir -p "${DATASET_DIR}"

if [[ "${RANGPUR}" == true ]]; then
  echo "Copying shared dataset from ${MAGENTA}${RANGPUR_SRC}${RESET}"
  cp -r "${RANGPUR_SRC}"/* "${DATASET_DIR}/"
  zips=("${RANGPUR_ZIPS[@]}")
else
  zips=("${ALL_ZIPS[@]}")
fi

for name in "${zips[@]}"; do
  download_and_unzip "${name}"
done
