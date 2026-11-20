#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
research_dir="$(cd "$script_dir/../.." && pwd)"
external_dir="$research_dir/baselines/external"
mkdir -p "$external_dir"

clone_at_commit() {
  local repository_url="$1"
  local destination="$2"
  local commit="$3"

  if [[ ! -d "$destination/.git" ]]; then
    git clone --filter=blob:none --no-checkout "$repository_url" "$destination"
  fi
  git -C "$destination" fetch --depth 1 origin "$commit"
  git -C "$destination" checkout --detach "$commit"
}

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

clone_at_commit \
  "https://github.com/cruiseresearchgroup/CGM-JEPA.git" \
  "$external_dir/CGM-JEPA" \
  "ef18c28517ce103f45069dec2594e3774b0fda41"
clone_at_commit \
  "https://github.com/Guylu/GluFormer.git" \
  "$external_dir/GluFormer" \
  "0bdaed9f16e0b9c7172c6fe5573e7e15f8f0ecac"
clone_at_commit \
  "https://github.com/Super-LzzZ/CGMformer.git" \
  "$external_dir/CGMformer" \
  "6a1c7ffcb738ab76f26195c6d39e4e080a31208b"
clone_at_commit \
  "https://github.com/IrinaStatsLab/GlucoBench.git" \
  "$external_dir/GlucoBench" \
  "661d840a98b316df51faa13a7100430afcbbb5b7"
clone_at_commit \
  "https://github.com/aametwally/Metabolic_Subphenotype_Predictor.git" \
  "$external_dir/Metabolic_Subphenotype_Predictor" \
  "94d647944c001dfa34492fd593b6e4804c1f45c4"

cgm_jepa_dir="$external_dir/CGM-JEPA"
model_base="https://huggingface.co/CRUISEResearchGroup/CGM-JEPA/resolve/468f6a4d721e6dcc1453913d8f7c41d24f5cef1a"
downstream_base="https://huggingface.co/datasets/CRUISEResearchGroup/CGM-JEPA-Downstream/resolve/a8312f611d736bf1be93e312570d7db6d4d669df"
pretrain_base="https://huggingface.co/datasets/CRUISEResearchGroup/CGM-JEPA-Pretraining/resolve/28d14baa61bd2eef9a92d10a9aa4739cd8108394"

for relative_path in \
  cgm_jepa/README.md \
  cgm_jepa/config.json \
  cgm_jepa/model.safetensors \
  x_cgm_jepa/README.md \
  x_cgm_jepa/config.json \
  x_cgm_jepa/model.safetensors \
  baselines/gluformer.pt \
  baselines/ts2vec.pkl; do
  download "$model_base/$relative_path" "$cgm_jepa_dir/Output/$relative_path"
done

for relative_path in \
  train.parquet \
  train_split.json \
  validation.parquet \
  validation_split.json; do
  download "$downstream_base/$relative_path" "$cgm_jepa_dir/Dataset_Open/$relative_path"
done
download \
  "$pretrain_base/cgm_initial_cohort.csv" \
  "$cgm_jepa_dir/Dataset_Open/cgm_initial_cohort.csv"

echo "Pinned baseline sources and CGM-JEPA artifacts are ready under $external_dir"
