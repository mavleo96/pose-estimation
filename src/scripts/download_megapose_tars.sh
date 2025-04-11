export LOCAL_DIR=/data/vijay/pose-estimation/datasets/gso/train_pbr_web
export REPO_ID=bop-benchmark/megapose

mkdir -p $LOCAL_DIR

for i in $(seq -f "%06g" 0 149); do
    FILE="MegaPose-GSO/shard-${i}.tar"
    echo "Downloading $FILE"

    retries=5
    count=0
    until huggingface-cli download $REPO_ID \
        --include "$FILE" \
        --local-dir $LOCAL_DIR \
        --repo-type=dataset || [ $count -eq $retries ]; do
        echo "Retrying ($count)..."
        sleep 5
        count=$((count + 1))
    done
done
