#!/bin/bash
# Exit immediately if any command exits with a non-zero status
set -e

# Default pattern to look for datasets

COMPETITORES="$(cat competitors.txt)"
DATASETS="$(cat datasets.txt)"

echo "=================================================="
echo "Competitors: $COMPETITORES"
echo "Datasets: $DATASETS"
echo "=================================================="

for COMPETITOR in $COMPETITORES; do 
        # here the COmpetitor end with \n, remove it
        COMPETITOR=${COMPETITOR%\n}
        #lowercase compet, remove / and ..
        COMPETITOR_L=${COMPETITOR//\/}
        COMPETITOR_L=${COMPETITOR_L//..}
        COMPETITOR_L=$(echo "$COMPETITOR_L" | tr '[:upper:]' '[:lower:]')
        just build-container auto-$COMPETITOR_L $COMPETITOR
done


echo -e $DATASETS

# Loop through each dataset and run evaluator
for DATASET in $DATASETS;  do
    # Remove trailing \r and whitespace
    DATASET=$(echo "$DATASET" | tr -d '\r' | xargs)
    if [ -z "$DATASET" ]; then
        continue
    fi
    echo "Running evaluation on: $DATASET"

    for COMPETITOR in $COMPETITORES; do 
        # Remove trailing \r and whitespace
        COMPETITOR=$(echo "$COMPETITOR" | tr -d '\r' | xargs)
        if [ -z "$COMPETITOR" ]; then
            continue
        fi
        
        COMPETITOR_L=${COMPETITOR//\/}
        COMPETITOR_L=${COMPETITOR_L//..}
        COMPETITOR_L=$(echo "$COMPETITOR_L" | tr '[:upper:]' '[:lower:]')
            
        
        python3 evaluator.py evaluate \
            --team ANNvedi \
            --image auto-$COMPETITOR_L \
            --dataset $DATASET
            
    done
done
