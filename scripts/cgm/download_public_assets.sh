#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
research_dir="$(cd "$script_dir/../.." && pwd)"
raw_dir="$research_dir/data/raw"
papers_dir="$research_dir/papers"

mkdir -p "$raw_dir" "$papers_dir"

download() {
  local url="$1"
  local destination="$2"
  if [[ -s "$destination" ]]; then
    return
  fi
  mkdir -p "$(dirname "$destination")"
  curl --location --fail --retry 3 --retry-delay 2 --continue-at - \
    --output "$destination" "$url"
}

# Paper: pin the current arXiv v2 PDF used by this research note.
download \
  "https://arxiv.org/pdf/2605.30865v2" \
  "$papers_dir/glucofm_v2_2605.30865.pdf"

# ShanghaiT1DM/T2DM, Figshare article version 5, CC BY 4.0.
shanghai_dir="$raw_dir/shanghai"
download \
  "https://ndownloader.figshare.com/files/42966622" \
  "$shanghai_dir/diabetes_datasets_v5.zip"
actual_md5="$(python -c 'import hashlib,sys; print(hashlib.md5(open(sys.argv[1], "rb").read()).hexdigest())' "$shanghai_dir/diabetes_datasets_v5.zip")"
if [[ "$actual_md5" != "4bfb61cfa506b48155fd9e841ee48e21" ]]; then
  echo "Shanghai archive MD5 mismatch" >&2
  exit 1
fi
unzip -n -q "$shanghai_dir/diabetes_datasets_v5.zip" -d "$shanghai_dir"

# Colás 2019 raw CGM series and reference DFA implementation, CC BY 4.0.
colas_dir="$raw_dir/colas2019"
download \
  "https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0225817.s001" \
  "$colas_dir/S1_raw_cgm.zip"
download \
  "https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0225817.s002" \
  "$colas_dir/S2_DFA_reference.R"
unzip -n -q "$colas_dir/S1_raw_cgm.zip" -d "$colas_dir"

# Hall/Glucotypes public phenotype files from Stanford.
hall_dir="$raw_dir/hall"
download \
  "https://web.stanford.edu/group/genetics/cgmdb/data/data_cgm.csv" \
  "$hall_dir/data_cgm.csv"
download \
  "https://web.stanford.edu/group/genetics/cgmdb/data/data_lipids.csv" \
  "$hall_dir/data_lipids.csv"
download \
  "https://web.stanford.edu/group/genetics/cgmdb/data/data_meta.csv" \
  "$hall_dir/data_meta.csv"

# BIG IDEAs 1.1.2: GlucoFM uses CGM only, so avoid the 34.1 GB full modality dump.
big_ideas_dir="$raw_dir/big_ideas_1.1.2"
download \
  "https://physionet.org/files/big-ideas-glycemic-wearable/1.1.2/Demographics.csv" \
  "$big_ideas_dir/Demographics.csv"
download \
  "https://physionet.org/files/big-ideas-glycemic-wearable/1.1.2/LICENSE.txt" \
  "$big_ideas_dir/LICENSE.txt"
download \
  "https://physionet.org/files/big-ideas-glycemic-wearable/1.1.2/SHA256SUMS.txt" \
  "$big_ideas_dir/UPSTREAM_SHA256SUMS.txt"
for subject_number in $(seq 1 16); do
  subject_id="$(printf '%03d' "$subject_number")"
  download \
    "https://physionet.org/files/big-ideas-glycemic-wearable/1.1.2/$subject_id/Dexcom_$subject_id.csv" \
    "$big_ideas_dir/$subject_id/Dexcom_$subject_id.csv"
done

find "$raw_dir" -type f ! -name SHA256SUMS.txt -print0 \
  | sort -z \
  | xargs -0 shasum -a 256 > "$raw_dir/SHA256SUMS.txt"

echo "Public research assets are ready under $raw_dir"

