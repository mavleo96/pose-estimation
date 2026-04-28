#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <target_directory> <output_file>"
    exit 1
fi

TARGET_DIR="$1"
OUTPUT="$2"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: '$TARGET_DIR' is not a valid directory"
    exit 1
fi

# Convert to absolute path
TARGET_DIR="$(realpath "$TARGET_DIR")"

> "$OUTPUT"

find "$TARGET_DIR" -type f -name "*.py" \
  ! -path "*/__pycache__/*" \
  ! -path "*/venv/*" \
  ! -path "*/.venv/*" | while read -r file; do

    echo "file_name: $file" >> "$OUTPUT"
    echo "start" >> "$OUTPUT"
    cat "$file" >> "$OUTPUT"
    echo "" >> "$OUTPUT"
    echo "end" >> "$OUTPUT"
    echo "" >> "$OUTPUT"
done

echo "Done → $OUTPUT"
