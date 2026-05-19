# On-the-Fly Fine-Grained Sketch-Based Image Retrieval with Dual-Channel Contrastive Learning and Stroke Self-Attention 

![](images/demo.png)
Illustration of our method’s capability to retrieve the target photo (top-10 list) with fewer strokes compared to SOTA methods such as RL-B, MGAL, LGRL and PSRL. The term T denotes the number of strokes.

## Framework
![](images/model.png)
(1) Phase 1 – FG-SBIR model based on Augmented-view contrastive, using NT-Xent Loss and Triplet Margin Loss. (2) The Stroke Self-Attention is used to learn the relational features among incomplete strokes. The weights of the other part are frozen

## Datasets
QMUL-Shoe-V2 and QMUL-Chair-V2 dataset was used.

## Kaggle ViT Training

The current default encoder is torchvision `ViT-B/16` with official
`IMAGENET1K_V1` weights. Set `--vit_variant b32` to use torchvision
`ViT-B/32` with official `IMAGENET1K_V1` weights instead.

```bash
git clone https://github.com/HieuNM1804/Onthefly.git
cd Onthefly
pip install -q -r requirements.txt
```

If Kaggle gives you a Tesla P100, install a PyTorch CUDA 11.8 build before
running the training cells. Newer Kaggle PyTorch builds may only include
`sm_70+` kernels, while P100 requires `sm_60`.

```bash
pip uninstall -y torch torchvision torchaudio
pip install -q torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

Restart the Kaggle session/kernel after reinstalling PyTorch, then verify:

```python
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
print(torch.cuda.get_arch_list())
```

Set the dataset root to the directory that contains `ChairV2` or `ShoeV2`:

```bash
ROOT_DIR=/kaggle/input/datasets/b20dccn616nguynhutun/fg-sbir-dataset
SAVE_DIR=/kaggle/working
DATASET_NAME=ChairV2
```

### Stage 1: baseline Triplet loss

This matches `on-the-fly-resnet.ipynb`: train the baseline for 200 epochs
without InfoNCE. `--use_info False` and `--alpha 1` make the loss pure
Triplet Margin Loss.

```bash
python main_baseline.py \
  --dataset_name ${DATASET_NAME} \
  --backbone ViT \
  --vit_variant b16 \
  --vit_pool cls_mean \
  --root_dir ${ROOT_DIR} \
  --save_dir ${SAVE_DIR} \
  --batch_size 64 \
  --epochs 200 \
  --use_info False \
  --alpha 1 \
  --backbone_lr 0.00001 \
  --lr 0.0001
```

### Stage 1.5: fine-tune Triplet + InfoNCE

This matches `on-the-fly-resnet-2.ipynb`: load the best baseline checkpoint
and fine-tune for 25 epochs with InfoNCE enabled. In `baseline/utils.py`,
the loss is:

```text
total_loss = alpha * TripletLoss + (1 - alpha) * InfoNCE
```

With `--alpha 0.1`, this is 10% Triplet Loss and 90% InfoNCE.

```bash
python main_baseline.py \
  --dataset_name ${DATASET_NAME} \
  --backbone ViT \
  --vit_variant b16 \
  --vit_pool cls_mean \
  --root_dir ${ROOT_DIR} \
  --save_dir ${SAVE_DIR} \
  --batch_size 28 \
  --load_pretrained True \
  --pretrained_dir ${SAVE_DIR}/${DATASET_NAME}_best_top10.pth \
  --epochs 25 \
  --use_info True \
  --alpha 0.1 \
  --backbone_lr 0.000001 \
  --lr 0.00001
```

### Stage 2: Stroke Self-Attention Triplet loss

This matches `on-the-fly-resnet-2-stage-2.ipynb`: load the stage 1.5 ViT
checkpoint parts and train only the Stroke Self-Attention module on partial
sketches. This stage uses `TripletMarginLoss` in `phase2/train.py`; it does
not use `--use_info`.

```bash
python main_sketch.py \
  --dataset_name ${DATASET_NAME} \
  --backbone ViT \
  --vit_variant b16 \
  --vit_pool cls_mean \
  --root_dir ${ROOT_DIR} \
  --save_dir ${SAVE_DIR} \
  --pretrained_dir ${SAVE_DIR} \
  --batch_size 48 \
  --epochs 100 \
  --steps 20
```

For ViT, prefer `b16` over `b32` for sketches. `b32` only gives a 7x7 patch
grid at 224px, which is usually too coarse for fine-grained sketch details.
