#!/bin/bash
# To make sure you can run this, you need to install the following:
# - s3cmd
# - azcopy
# and configure them with the necessary credentials
# s3cmd --configure
# azcopy login

# Ensure script execution stops on error
set -e

# Parameters
SONAR_NAME=$1
YEAR=$2
MONTH=$3
START_DAY=$4
END_DAY=$5
STORAGE_TIER=${6:-cool}

# Base directories
TEMP_DIR="./temp_videos"
S3_BUCKET="sil-assainissementpiscicole"
AZURE_CONTAINER="https://axe4lab4appl4fishy4sa.blob.core.windows.net/azureml-blobstore-1a903f0a-371e-458d-b4e6-fcc92fbf93c8"
ENDPOINT_URL="https://sos-ch-gva-2.exo.io"

# Process each day within the day range
for (( DAY=$START_DAY; DAY<=$END_DAY; DAY++ ))
do
    # Format day for leading zero
    FORMATTED_DAY=$(printf "%02d" $DAY)
    
    # Define directory paths
    S3_PATH="s3://${S3_BUCKET}/${SONAR_NAME}/${YEAR}/${MONTH}/${FORMATTED_DAY}/"
    LOCAL_DIR="${TEMP_DIR}/${SONAR_NAME}/${YEAR}/${MONTH}/${FORMATTED_DAY}/"
    AZURE_PATH="${AZURE_CONTAINER}/lavey_videos/${SONAR_NAME}/${YEAR}/${MONTH}/"
    
    # Create local directory
    mkdir -p $LOCAL_DIR
    
    # Download from S3 to local directory
    s3cmd get --recursive $S3_PATH $LOCAL_DIR --host=$ENDPOINT_URL
    
    # Upload from local directory to Azure
    azcopy copy "$LOCAL_DIR" "$AZURE_PATH" --recursive --block-blob-tier=$STORAGE_TIER
    
    # Remove local directory contents securely
    rm -rf $LOCAL_DIR
done

# Cleanup: Ensure the base temporary directory is also removed after processing
rm -rf "$TEMP_DIR/${SONAR_NAME}"

echo "Data transfer complete."
