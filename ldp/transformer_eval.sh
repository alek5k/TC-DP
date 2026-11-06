#!/bin/bash

# Parse flagged arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --checkpoint) ckpt="$2"; shift ;;
        --name) name="$2"; shift ;;
        --perturb) perturb="$2"; shift ;;
        --samples) samples="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 --checkpoint PATH --name NAME --perturb VALUE --samples N"
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [[ -z "$ckpt" || -z "$name" || -z "$perturb" || -z "$samples" ]]; then
    echo "Missing required arguments."
    echo "Run with -h or --help to see usage."
    exit 1
fi

hhmmss=$(date +"%H%M%S")
export MUJOCO_GL="egl"

python eval.py \
    --checkpoint "$ckpt" \
    -o "data/${name}_$hhmmss" \
    -p "$perturb" \
    -n "$samples"

