#!/usr/bin/env bash
# Downloads the Kaggle Chest X-Ray Pneumonia dataset into data/chest_xray/
# Needs a Kaggle API token: kaggle.com -> Settings -> API -> Create New Token (kaggle.json)
set -e
mkdir -p ~/.kaggle data
if [ -f kaggle.json ]; then cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json; fi

kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data
unzip -q -o data/chest-xray-pneumonia.zip -d data
rm -rf data/chest_xray/__MACOSX data/chest_xray/chest_xray   # duplicate nested copy in the zip
rm -f data/chest-xray-pneumonia.zip

echo "Done. Image counts:"
for s in train val test; do
  for c in NORMAL PNEUMONIA; do
    echo "  $s/$c: $(ls data/chest_xray/$s/$c | wc -l)"
  done
done
