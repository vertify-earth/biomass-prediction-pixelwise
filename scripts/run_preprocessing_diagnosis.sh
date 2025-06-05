#!/bin/bash
# Script to run the biomass preprocessing diagnostic tool
# Date: 2025-05-17
# Author: najahpokkiri

IMAGE_PATH="/teamspace/studios/dl2/clean/data/s1_s2_l8_palsar_ch_goa_goa_2020.tif"
OUTPUT_DIR="biomass_preprocessing_diagnosis_$(date +%Y%m%d_%H%M%S)"
CHUNK_SIZE=1024
OVERLAP=32

echo "===== Biomass Preprocessing Diagnostic Tool (Fixed Version) ====="
echo "Date: $(date)"
echo "Input image: $IMAGE_PATH"
echo "Output directory: $OUTPUT_DIR"

# Make sure requirements are installed
echo "Installing requirements..."
pip install -q numpy matplotlib torch rasterio scikit-learn scikit-image

# Run the diagnostic tool
echo "Running diagnostic analysis..."
python diagnose_biomass_preprocessing.py "$IMAGE_PATH" --output-dir "$OUTPUT_DIR" --chunk-size "$CHUNK_SIZE" --overlap "$OVERLAP"

echo "Analysis complete. Check the $OUTPUT_DIR directory for results."