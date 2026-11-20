# Public data and evaluation protocol

## Sources

| Dataset | Use | Upstream source |
|---|---|---|
| CGMacros | Four downstream tasks | [PhysioNet CGMacros](https://physionet.org/content/cgmacros/) |
| ShanghaiT2DM | Disjoint pretraining subset and three downstream tasks | Figshare diabetes dataset version 5, file ID `42966622`; downloader pins archive MD5 |
| Stanford | Disjoint pretraining subset and three downstream tasks | `aametwally/Metabolic_Subphenotype_Predictor`, commit `94d647944c001dfa34492fd593b6e4804c1f45c4` |
| Hall | Four downstream tasks | `IrinaStatsLab/GlucoBench`, commit `661d840a98b316df51faa13a7100430afcbbb5b7`, `raw_data/hall.csv` |
| BIG IDEAs | Unlabeled pretraining | PhysioNet `big-ideas-glycemic-wearable/1.1.2`, CGM files only |
| Colás 2019 | Unlabeled pretraining | PLOS ONE `10.1371/journal.pone.0225817`, S1 supplement |

Download links and pinned upstream code revisions are in `scripts/cgm/download_public_assets.sh` and `scripts/cgm/bootstrap_baselines.sh`. They access only public upstream services and do not log in to private infrastructure. Access conditions and redistribution rights remain those of the original providers; no access restrictions are bypassed. Upstream availability can change.

## Tasks

- CGMacros: diabetes, insulin resistance, hyperlipidemia, obesity.
- ShanghaiT2DM: insulin resistance, hyperlipidemia, hypoglycemia.
- Stanford: diabetes, insulin resistance, beta-cell dysfunction.
- Hall: diabetes, insulin resistance, hyperlipidemia, glucotype.

Exact label thresholds, eligibility, native-reading recovery, missingness handling, identity grouping, and metrics are implemented in `lingsante_pulse.cgm.data.protocol` and the four dataset builders. Inspect each builder's output report before evaluation. The NPZ key names are the authoritative interface; do not substitute labels constructed with different thresholds.

The canonical input is a non-overlapping 24-hour span with 288 five-minute grid positions. Missing observations remain masked even when interpolation is used internally. CGMacros native anchors are recovered from its interpolated release. Episode boundaries prevent context from crossing large recording gaps. Longitudinal inference uses only the same subject/episode's available current and preceding windows.

The public pretraining builder separates Shanghai biological identities where available and Stanford's pretraining identities from its downstream set. Downstream Shanghai contains 65 visits from 58 biological participants; `lingsante_pulse.cgm.evaluation.protocol` joins visits using the same participant prefix as the original builder before assigning folds. All visits of one participant remain together in both inner and outer partitions. CGMacros, Stanford, and Hall contain 45, 37, and 56 evaluation participants. Pretraining split 0 is used for fitting, split 1 for unlabeled validation, and private held-out split 2 is excluded. The pretraining episode sampler uses recorded identities, whose detailed scope is specified in the manuscript appendix.

## Private source used by the supplied weights

The supplied model combines components trained on stage-specific unlabeled corpora. Its longitudinal context and memory used 8,421 windows (6,138 private and 2,283 public); its final daily branch used 9,956 windows (7,673 private and 2,283 public); and its final dynamics branch used 13,025 windows (10,742 private and 2,283 public). The private windows were selected separately across stages, so these are neither additive nor nested dataset sizes. The 8,421-window corpus had 7,341 fitting and 1,080 validation windows; 649 private held-out windows were excluded. Private phenotype labels were not supplied to representation training. All four public downstream cohorts remain external evaluation datasets under the documented protocol. The private source's name, organization, raw records, subject identifiers, and access routes are not part of the release.

The public-only training recipe does not require this private source. The pretraining NPZ is not the downstream evaluation NPZ: never substitute all downstream subjects as unlabeled training inputs simply because their phenotype labels have been removed.

## Metrics

Evaluation uses `lingsante_pulse.cgm.evaluation.protocol.common_folds`: deterministic balanced-size, biological-person partitions with class coverage shared across all tasks of a cohort. It fits `StandardScaler + LogisticRegression(C=1, L2, lbfgs, max_iter=1000)` within each training partition. PR-AUC is average precision, with macro one-versus-rest averaging for multiclass tasks; multiclass ROC-AUC is also macro one-versus-rest. Macro-F1 uses the highest-probability class. Scores average folds within task, then weight the 14 tasks equally. The 10 repeats × 5 folds are evaluation partitions, not additional pretraining seeds. `lingsante_pulse.cgm.evaluation.nested` adds training-only three-fold candidate selection; `lingsante_pulse.cgm.evaluation.fixed` evaluates fixed models on the same outer partitions. Published baseline scores retain their source-study settings.
