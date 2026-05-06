# CV Final Project — OCRNet (ResNet-50) on ADE20K

This fork adds scripts for training OCRNet on the MIT Engaging cluster.

### Repository layout

| Path | Purpose |
|------|---------|
| [`mmseg/`](mmseg/) | Library source: training loop, models (including OCRNet), datasets. |
| [`configs/`](configs/) | Training configs; this fork keeps OCRNet and FCN on ADE20K (20% and 50% splits). |
| [`metadata/`](metadata/) | `model-index.yml` and `dataset-index.yml` for MIM / dataset tooling (installed into `mmseg/.mim/`). |
| [`tools/`](tools/) | CLI entry points (`train.py`, `test.py`, etc.). |
| [`setup_cluster.sh`](setup_cluster.sh), [`train_slurm_ocr.sh`](train_slurm_ocr.sh), [`train_slurm_fcn.sh`](train_slurm_fcn.sh), [`train_slurm_50pct.sh`](train_slurm_50pct.sh) | Engaging cluster environment setup and SLURM jobs. |
| `data/` | Datasets (gitignored). Place ADE20K under `data/ade/` as in step 4. |
| `work_dirs/`, `logs/` | Checkpoints and job logs (gitignored / created at runtime). |

## 1. SSH into Engaging

If you don't already have an account, log in once at https://orcd-ood.mit.edu with your MIT Kerberos credentials to create one.

```bash
ssh YOUR_KERBEROS@orcd-login.mit.edu
```

## 2. Clone this repo

```bash
cd ~
git clone https://github.com/mag-zhou/cvfinalproj-ocrnet.git mmsegmentation
cd mmsegmentation
```

## 3. Set up the conda environment

Run the setup script — this creates a `mmseg` conda env (Python 3.11) and installs PyTorch + OpenMMLab dependencies. Takes ~5–10 minutes.

```bash
bash setup_cluster.sh
```

Then install one extra dependency:

```bash
module load miniforge
source /orcd/software/core/001/pkg/miniforge/25.11.0-0/etc/profile.d/conda.sh
conda activate mmseg
pip install ftfy regex "numpy<2"
```

Verify the install:

```bash
python -c "import mmengine, mmcv, mmseg; print('mmengine', mmengine.__version__); print('mmcv', mmcv.__version__); print('mmseg', mmseg.__version__)"
```

## 4. Download ADE20K

```bash
mkdir -p data/ade
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip -P data/ade/
unzip data/ade/ADEChallengeData2016.zip -d data/ade/
rm data/ade/ADEChallengeData2016.zip
```

Final structure should be:
```
data/ade/ADEChallengeData2016/
├── images/{training,validation}/
└── annotations/{training,validation}/
```

## 5. Train OCRNet (ResNet-50)

The provided config at `configs/ocrnet/ocrnet_r50-d8_1xb8-40k_ade20k-512x512-20pct.py` trains on the first 20% of ADE20K (4,042 images) with batch size 8, learning rate 0.002, for 40,000 iterations.

Submit the SLURM job for ocrnet:

```bash
sbatch train_slurm_ocr.sh
```

Monitor:

```bash
squeue -u $USER                  # check job status (PD=pending, R=running)
tail -f logs/train_<JOBID>.out   # live training log
ls work_dirs/ocrnet_r50_ade20k_20pct/   # checkpoints saved here
```

Estimated training time on an L40S GPU: **~3–4 hours**.

## 6. Updating the cluster with local changes

Edit configs locally → push → pull on cluster:

```bash
# Locally
git add <files> && git commit -m "..." && git push origin main

# On cluster
cd ~/mmsegmentation && git pull origin main
```

## 7. FCN baseline (optional)

```bash
sbatch train_slurm_fcn.sh
```

Checkpoints: `work_dirs/fcn_r50_ade20k_20pct/`

## 8. 50% ADE20K experiment (OCRNet)

Config: `configs/ocrnet/ocrnet_r50-d8_1xb8-40k_ade20k-512x512-50pct.py`  
Hyperparams: 10,105 images, batch size 8, lr 0.004, 40k iterations, checkpoint every 2,000 iters.

Submit the job:

```bash
git pull origin main   # make sure cluster has latest configs
sbatch train_slurm_50pct.sh
```

Monitor:

```bash
squeue -u $USER                           # check job status
tail -f logs/train_<JOBID>.out            # live training log
ls work_dirs/ocrnet_r50_ade20k_50pct/    # checkpoints saved here (last 3 kept)
```

Estimated training time on an L40S GPU: **~3–4 hours** (same iteration count as 20pct run).

### Resuming if the job times out

The script `train_slurm_50pct.sh` includes automatic resume logic. If the job runs out of time before finishing, simply resubmit the same script — it will detect the last saved checkpoint and continue from there:

```bash
sbatch train_slurm_50pct.sh
```

To confirm the new job resumed correctly (rather than starting over):

```bash
grep "Resumed from" logs/train_<NEW_JOBID>.out
```

You should see a line like `Resumed from: work_dirs/ocrnet_r50_ade20k_50pct/iter_XXXXX.pth`.

### Checking final results

```bash
grep "mIoU" work_dirs/ocrnet_r50_ade20k_50pct/*/*.log | tail -20
```

---

## Upstream MMSegmentation

This repository is a fork of [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Most extra config families, in-repo `docs/`, `demo/`, and `projects/` were removed to keep the tree small. The full model zoo and tutorials are in the upstream repository. API reference: [mmsegmentation.readthedocs.io](https://mmsegmentation.readthedocs.io/en/latest/).
