#!/bin/bash

SI_DIR="$HOME/.local/share/SI"

mkdir -p "$SI_DIR"

model_url="https://huggingface.co/GunA-SD/Hub/resolve/main/SI/model.bin?download=true"
tokenizer_url="https://huggingface.co/GunA-SD/Hub/resolve/main/SItokenizer.bin?download=true"

model_path="$SI_DIR/model.bin"
tokenizer_path="$SI_DIR/tokenizer.bin"

echo "Downloading model..."
if curl --fail --silent --show-error --output "$model_path" "$model_url"; then
    echo "Model downloaded successfully."
else
    echo "Failed to download model. Please check Internet connection."
    exit 1
fi

echo "Downloading tokenizer..."
if curl --fail --silent --show-error --output "$tokenizer_path" "$tokenizer_url"; then
    echo "Tokenizer downloaded successfully."
else
    echo "Failed to download tokenizer. Please check Internet connection."
    exit 1
fi