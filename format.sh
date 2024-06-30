#!/bin/bash
SOURCE_DIRS=("Sources" "Tests")
FORMAT_CONFIG="format.json"

if ! command -v swift-format &> /dev/null
then
    echo "swift-format could not be found. Please install it first."
    exit 1
fi

if [ ! -f "$FORMAT_CONFIG" ]; then
    echo "Format configuration file $FORMAT_CONFIG does not exist."
    exit 1
fi

for DIR in "${SOURCE_DIRS[@]}"; do
    if [ ! -d "$DIR" ]; then
        echo "Source directory $DIR does not exist."
        continue
    fi
    find "$DIR" -type f -name "*.swift" -print0 | while IFS= read -r -d '' file; do
        echo "$file"
        swift-format format --configuration "$FORMAT_CONFIG" "$file" -i
    done
done

echo "Formatting completed."
