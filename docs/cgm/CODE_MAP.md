# Code map

## CGM workflows

| Purpose | Entry or module |
|---|---|
| End-to-end public-only pretraining | `lingsante_pulse.cgm.training.public` |
| Daily common/dense-only encoder training | `lingsante_pulse.cgm.training.daily`, `lingsante_pulse.cgm.models.daily` |
| Signed-dynamics training | `lingsante_pulse.cgm.training.dynamics`, `lingsante_pulse.cgm.models.dynamics` |
| Physical scale parameters | `lingsante_pulse.cgm.training.assemble_scale` |
| Context-stage training | `lingsante_pulse.cgm.training.context` |
| Final transferable-evidence training | `lingsante_pulse.cgm.training.memory` |
| Final representation module | `lingsante_pulse.cgm.models.memory` and its inherited context modules |
| Evidence and observation support | `lingsante_pulse.cgm.models.final_space_longitudinal`, `lingsante_pulse.cgm.models.longitudinal_evidence_encoder`, `lingsante_pulse.cgm.models.episodic_evidence` |
| Portable checkpoint inference | `lingsante_pulse.cgm.inference.pipeline` |
| Training-to-inference checkpoint export | `lingsante_pulse.cgm.checkpoints` |
| Optional 24-candidate nested evaluation | `lingsante_pulse.cgm.evaluation.nested`, `lingsante_pulse.cgm.evaluation.protocol`, a separately supplied candidate bundle |
| Fixed-model evaluation on the same person folds | `lingsante_pulse.cgm.evaluation.fixed` |

## Baseline code

GlucoFM is an independent reproduction, not a component of the proposed encoder. Public-only pretraining example:

```bash
python -m lingsante_pulse.cgm.baselines.train_glucofm \
  --npz data/processed/public_pretraining/glucofm_public_pretrain_subject_disjoint_v1.npz \
  --output-dir results/glucofm_reproduction --device cuda:0 \
  --epochs 120 --batch-size 128 --seed 43 \
  --relative-auxiliary-features --balance-cohort-cadence

python -m lingsante_pulse.cgm.baselines.glucofm \
  --npz data/processed/glucofm_public/hall/hall_glucofm_windows_v1.npz \
  --checkpoint results/glucofm_reproduction/best.pt \
  --output results/glucofm_hall_embeddings.npz --device cuda:0
```

The GlucoFM baseline is a local implementation based on the paper. It is not the upstream official implementation or checkpoint.

Released CGM-JEPA / X-CGM-JEPA checkpoints:

```bash
python -m lingsante_pulse.cgm.baselines.jepa \
  --npz data/processed/glucofm_public/hall/hall_glucofm_windows_v1.npz \
  --reference-corpus data/processed/public_pretraining/glucofm_public_pretrain_subject_disjoint_v1.npz \
  --baseline-root baselines/external/CGM-JEPA --model x_cgm_jepa \
  --output results/x_cgm_jepa_hall_embeddings.npz --device cuda:0
```

Use `--model cgm_jepa` for the other released checkpoint. Their upstream encoder requires its upstream dependencies in addition to this package's requirements. `bootstrap_baselines.sh` pins upstream revisions and weight URLs; third-party weights are not bundled.

CGMformer uses `lingsante_pulse.cgm.baselines.cgmformer`; its required arguments are `--npz`, `--baseline-root`, `--checkpoint`, `--token-dictionary`, and `--output`. Use the released model directory and token dictionary obtained from `Super-LzzZ/CGMformer` at the pinned revision. The default `--protocol released_author_code` preserves the published author's pooling/masking implementation. It requires `transformers`; do not replace its original tokenizer with a generic tokenizer. The alternative protocol flag is not the main comparison.

Arrange each baseline's four cohort embedding files under `results/baseline/COHORT/embeddings.npz`, then apply the common person-grouped evaluator:

```bash
python -m lingsante_pulse.cgm.evaluation.fixed \
  --data-root data/processed/glucofm_public \
  --embeddings-root results/baseline --output results/baseline_evaluation
```

