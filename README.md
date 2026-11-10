# CSSL: contrastive semi-supervised segmentation

Training and evaluation code for CSSL on MoNuSeg (50% labels, Self-Loop's protocol) and CHAOS T2 (25% labels,
DAFNet's protocol), with the recipe-only and supervised baselines and all ablations of the paper.

| | MoNuSeg 50% (nuclei F1) | CHAOS T2 25% (Dice) |
|---|---|---|
| CSSL | 80.87 ± 0.17 (seeds 0, 1, 2: 80.66, 81.07, 80.88) | 89.01 ± 2.37 (splits 0, 1, 2: 90.47, 85.67, 90.90) |
| Recipe only | 78.73 ± 0.95 | 88.94 ± 2.63 |
| Published baseline | Self-Loop 79.10 | DAFNet 82.0 |

## 1. Conda environment

```bash
conda create -y -n cssl python=3.10
conda activate cssl
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m pytest -q tests
```

Use `cu118` instead of `cu121` for older NVIDIA drivers. `bash setup/install_env.sh` runs the same steps.

## Data download and preparation by hand (optional)

`STAGES=setup` does all of this automatically. To do it by hand, for example when compute nodes have no internet:

```bash
export DATA=/data/med
mkdir -p $DATA/downloads && cd $DATA/downloads

wget -c "https://zenodo.org/records/3431873/files/CHAOS_Train_Sets.zip?download=1" -O CHAOS_Train_Sets.zip
md5sum CHAOS_Train_Sets.zip
mkdir -p chaos_raw && unzip -q CHAOS_Train_Sets.zip -d chaos_raw

gdown 1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA -O monuseg_train.zip
gdown 1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw -O monuseg_test.zip
mkdir -p monuseg_raw && unzip -q monuseg_train.zip -d monuseg_raw && unzip -q monuseg_test.zip -d monuseg_raw

cd /path/to/cssl-repro
python tools/prep_monuseg.py --src $DATA/downloads/monuseg_raw --out $DATA/monuseg
python tools/prep_chaos_dafnet.py --src $DATA/downloads/chaos_raw --out $DATA/chaos_dafnet
python tools/make_splits.py --dataset monuseg --root $DATA/monuseg
python tools/make_splits.py --dataset chaos --protocol dafnet --root $DATA/chaos_dafnet
```

`md5sum` must print `df21053002a1cc86df918a87da3b2c19`. If `gdown` is blocked, download the two MoNuSeg files in a
browser from `https://drive.google.com/file/d/1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA` (training) and
`https://drive.google.com/file/d/1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw` (test), save them as `monuseg_train.zip` and
`monuseg_test.zip` in `$DATA/downloads`, and continue with the `unzip` line. Expected result:

| Check | Expected |
|---|---|
| `ls $DATA/monuseg/train/images \| wc -l` | 30 |
| `ls $DATA/monuseg/test/images \| wc -l` | 14 |
| `ls $DATA/chaos_dafnet/*.npz \| wc -l` | 40 (20 subjects, T1 and T2) |
| `ls splits/monuseg splits/chaos` | `seed0.json seed1.json seed2.json`, `dafnet0.json dafnet1.json dafnet2.json` |

Overlays of images and labels for a visual check are written to `$DATA/monuseg/overlays` and
`$DATA/chaos_dafnet/overlays`. Then run the experiments without the setup stage: `STAGES="runs"`.

## 2. Run everything end to end

```bash
conda activate cssl
DATA=/data/med STAGES="setup runs" nohup bash run_paper.sh > run_paper.log 2>&1 &
tail -f run_paper.log
```

`DATA` is any folder for the datasets. The command downloads and prepares the data, trains all 28 runs (about
45 GPU-hours on one A100-80GB, 4 runs at a time), prints the result tables.

If a walltime limit stops it, run the same command again: finished runs are skipped and interrupted runs continue
from their last checkpoint. The three stages can also be run separately (`STAGES=setup`, `STAGES=runs`), and the datasets can be split over two GPUs:

```bash
DATA=/data/med DATASETS=monuseg STAGES=runs nohup bash run_paper.sh > mon.log 2>&1 &
DATA=/data/med DATASETS=chaos   STAGES=runs nohup bash run_paper.sh > chaos.log 2>&1 &
```

CSSL settings (added to `configs/monuseg_selfloop.yaml` and `configs/chaos_dafnet.yaml`):

| Dataset | Settings |
|---|---|
| MoNuSeg | `cssl.cr.mode=add cssl.cr.add_weight=0.5 cssl.cl.embed_from=image cssl.cl.embed_level=-3 cssl.cl.queue=4096 cssl.cl.boundary=2` |
| CHAOS | `cssl.cr.mode=add cssl.cr.add_weight=0.5` |

## 3. What the code does, step by step

**Stage `setup`** (`setup/prepare_data.sh`, `tools/prep_chaos_dafnet.py`, `tools/make_splits.py`)
1. Downloads CHAOS from Zenodo (checksum verified) and MoNuSeg from the challenge's Google Drive.
2. Converts MoNuSeg to image and mask files: 30 training and 14 test images.
3. Converts CHAOS as DAFNet's code does: T2-SPIR and T1-DUAL out-phase, DAFNet's slice ranges, 192 x 192.
4. Writes the splits: DAFNet's three CHAOS splits (14 / 3 / 3 subjects, 4 labelled T2 subjects) and three random
   MoNuSeg label subsets of 15 images (seeds 0, 1, 2).

**Stage `runs`** (`run_paper.sh` calls `train.py` once per run)
1. Stage 1, 26 runs:
   - MoNuSeg: CSSL and recipe only on seeds 0, 1, 2; supervised with 50% and 100% labels; 6 ablations of CSSL
     on seed 0 (without contrastive loss, consistency loss, hard negatives, boundary class, feature-level masking,
     memory bank).
   - CHAOS: CSSL, recipe only and supervised on splits 0, 1, 2; 3 ablations of CSSL on split 1 (without
     contrastive loss, consistency loss, hard negatives).
2. Stage 2, 2 runs: CSSL with a fixed lambda (MoNuSeg seed 0, CHAOS split 1), set to the average lambda of the
   corresponding CSSL run.
3. After each run it prints the run name, its test score and the progress, e.g. `[Stage 1/2: 12/26 runs done]`.

**Inside one run** (`train.py`)
1. Loads the configuration and the split, builds the network (ResUNet-18 for MoNuSeg, U-Net 64-1024 for CHAOS) and
   an EMA copy of it (the teacher).
2. Iterations 1-2000: supervised training on the labelled images (cross-entropy + Dice).
3. Every later iteration:
   - labelled batch: cross-entropy + Dice against the ground truth;
   - unlabelled batch: the teacher predicts on a weakly augmented view; pixels with confidence >= 0.95 become
     pseudo-labels;
   - two strongly augmented views with CutMix are trained on the pseudo-labels with Dice (this is the recipe);
   - consistency loss: Gaussian noise of strength lambda is added to all encoder features, and the prediction from
     the perturbed features is trained on the pseudo-labels (cross-entropy + Dice);
   - contrastive loss: 64 x 64 windows over the labels and pseudo-labels give one patch per class (MoNuSeg:
     nucleus, nucleus rim, background; CHAOS: the dominant organ); each patch is embedded (MoNuSeg: masked pooling
     of 1/8-resolution encoder features of the whole image; CHAOS: pixel-masked patch through the encoder) and a
     supervised contrastive loss with mixed hard negatives (and, on MoNuSeg, a memory bank of 4096 embeddings) is
     applied;
   - the Silhouette score omega of the patch embeddings sets the next noise level, lambda = (omega + 1) / 2 + 0.05;
   - SGD step on the total loss, then EMA update of the teacher.
4. Logs losses, lambda and omega to `metrics.jsonl` and saves a checkpoint for resuming (weights, optimizer,
   teacher, omega, memory bank and random-generator states), so an interrupted run continues where it stopped.
5. CHAOS: computes the Dice on the 3 validation subjects every 1000 iterations and keeps the best checkpoint.
6. At the end, tests the teacher network: MoNuSeg on the 14 whole test images with the final weights (nuclei F1);
   CHAOS on the 3 test subjects with the best-validation checkpoint (DAFNet's Dice formula on argmax masks; the
   score with DAFNet's per-organ rounding is stored as `dafnet_dice_round`). Writes `results.json` with the score of
   every test image or subject, and `final.pth`.


## 4. Outputs

| Item | Location |
|---|---|
| One run (configuration, log, metrics, test scores, weights) | `exp/<run>/<dataset>_s<split>/` |
| MoNuSeg table | `exp/main_sup`, `exp/mabl_none`, `exp/cssl_strongest`, `exp/sup100` |
| CHAOS table | `exp/main_sup`, `exp/cabl_none`, `exp/cssl_strongest` |
| Ablations | `exp/sabl_*` |
| Logs of all runs | `exp/paper_logs/` |


## 5. Protocols

| | MoNuSeg (Self-Loop) | CHAOS T2 (DAFNet) |
|---|---|---|
| Data | 30 training / 14 test images, 1000 x 1000 | 20 subjects, DAFNet's 3 splits (14 / 3 / 3) |
| Labelled | 15 images (3 random subsets, seeds 0-2) | 4 T2 subjects per split, T1 labels for all training subjects |
| Network | ResUNet-18, random initialisation | U-Net, 64-1024 channels |
| Metric | nuclei F1 (per-image nucleus Dice) | DAFNet's per-slice pooled Dice (argmax masks; rounded version also printed) |
| Tested weights | final | best validation Dice |

## 6. Pipeline test on synthetic data (CPU, about 30 minutes)

```bash
python tests/make_synthetic.py --out /tmp/syn --chaos-size 64 --chaos-slices 50 --chaos-t1-slices 50
python tools/prep_monuseg.py --src /tmp/syn/monuseg_raw --out /tmp/syn/data/monuseg
python tools/prep_chaos_dafnet.py --src /tmp/syn/chaos_raw --out /tmp/syn/data/chaos_dafnet --size 64 --allow-mismatch
python tools/make_splits.py --dataset monuseg --root /tmp/syn/data/monuseg
python tools/make_splits.py --dataset chaos --protocol dafnet --root /tmp/syn/data/chaos_dafnet
DATA=/tmp/syn/data DEVICE=cpu PAR=2 EXTRA_OPTS="total_iters=6 warmup_iters=2 eval_every=3 batch_size=2 \
  crop_size=64 num_workers=0 log_every=1 cssl.cl.rampup_iters=2 crops_per_image=2" bash run_paper.sh
```

Use a copy of the repository for this test, since it writes `splits/` and `exp/`.

## Layout

```
train.py       training and evaluation
cssl/          class-wise patches, contrastive loss, Silhouette, feature perturbation
model/         U-Net, ResUNet-18, CSSL network (projection head, feature-level masked pooling)
data/          datasets and augmentations
configs/       monuseg_selfloop.yaml, chaos_dafnet.yaml
tools/         data preparation, splits, audit, print_results, export_paper_numbers, qualitative, video_assets
setup/         install_env.sh, prepare_data.sh
tests/         unit tests, synthetic data
run_paper.sh   all stages
paper/         main.tex, references.bib, numbers.json, gen_numbers.py, make_figures.py, figures/, video/
```

Licenses: the U-Net, augmentations and Dice loss come from UniMatch (MIT, `LICENSE.UniMatch`). CHAOS and MoNuSeg are
CC BY-NC-SA 4.0.
