for file in $LOCAL_DIR/$NAME/*.zip; do
    if [[ $file == *_base.zip ]]; then
        unzip -q -j "$file" -d "$(dirname "$file")"
        echo "Extraction of base file $file is complete."
    elif [[ $file == *_all.zip ]]; then
        echo "Skipping $file as it is an _all.zip file."
    elif [[ $file == *bop19.zip ]]; then
        echo "Unzipping $file..."
        unzip -q "$file" -d "$(dirname "$file")"
        echo "Extraction of $file is complete."
    else
        echo "Unzipping $file..."
        unzip -q "$file" -d "$(dirname "$file")"
        echo "Extraction of $file is complete."
    fi
done