# AN1 Meaning Engine
64 Dimensional Frozen Sender

---

## Summary

This repo demonstrates a simple empirical result:

**A frozen network’s early-layer activations already contain a usable sketch of the final semantic intent.  
By decoding only the first block’s activation, we recover ~82.6 percent of the full model’s accuracy while skipping ~99.93 percent of the compute.**

There is no early exit, no pruning, no partial forward pass, and no distillation.  
**The backbone stops after block 1. Nothing else runs.**

In other words, the network’s intent sketch is already present after the first block, long before the deep stack performs its work.

---

## What This Shows (one sentence)

**The final intent of a frozen network is already mostly formed in the early layers, and a tiny learned decoder can reconstruct most of the teacher’s behavior.**

---

## What This Is *Not*

This method is **not**:

- early exit  
- pruning or sparsification  
- distillation  
- dynamic halting  
- low-rank compression  
- partial forward pass  

All of those require running many backbone layers.  
**This runs only block 1.**

---

## Core Experiment

- Freeze a ResNet18 teacher  
- Take the activation after **block 1**  
- Project it to **64 dimensions**  
- Train a small MLP head  
- Skip all deeper layers (~99.93 percent of FLOPs)  
- Measure accuracy  

**Result: ~82.6 percent accuracy recovery at ~1370× fewer FLOPs.**

This is the empirical point the repo demonstrates.

---

## Why It Matters

If early-layer vectors contain enough semantic intent to reconstruct most downstream prediction, then:

- deep stacks may be over-parameterized for inference  
- early activations behave more like compressed intent sketches than raw features  
- large portions of inference pipelines may be avoidable  

This repo provides a reproducible testbed for exploring that idea.

---

## Diagram

### Normal path
```
input → block1 → block2 → block3 → ... → blockN → head → prediction
```

### This repo
```
input → block1 → tiny decoder → prediction
(skip blocks 2..N)
```

---

## Results


**Teacher:** ResNet18 at 87.89 percent accuracy  
**Header:** 64 dimensional vector from early-layer output  
**AN1 Head:** Tiny MLP (64 -> 1024 -> 512 -> 128 -> 10)

### Outcome


**72.57 percent accuracy**  
**10.15x speedup** (wall clock per example, batch size 256)  
**1370.6x reduction** in floating point operations

**Speedup by batch size:**  
- Batch size 64: 8.87x

The implication is simple.
Even small signals can carry surprising intent.
This experiment makes that visible.

### Baselines

To separate the effect of pretrained early features from raw pixels or scratch training, the repo includes two matched baselines that use the same MLP head as AN1:

- **Pixels → MLP**  
- **Scratch block1 → MLP**  
- **Frozen ResNet block1 → AN1 head** (main experiment)

All three use identical head capacity and optimizer settings.
The only difference is the input representation.

The results (from an H200 NVL run) are stored in:

results/baselines.json

Reproduce with:

```bash
python -m an1_meaning_engine.baselines.experiment_pixels
python -m an1_meaning_engine.baselines.experiment_scratch_block1
python -m an1_meaning_engine.experiment_frozen_sender

H200 Results:

Model	Accuracy

Teacher (ResNet18 full)	87.89 %
Frozen block1 + AN1 head	72.57 %
Scratch block1 + same head	66.89 %
Pixels + same head	52.25 %

This cleanly shows that the frozen teacher’s first block contains more task-aligned information than either raw pixels or a scratch-trained early block, under identical head capacity and training time.

---

## How to Run

### Install dependencies
pip install -r requirements.txt

### Teacher checkpoint

This experiment uses a pretrained, frozen teacher checkpoint:
checkpoints/resnet18_cifar10_teacher.pth

The checkpoint is already included.  
You only need to retrain it if you intentionally want a new teacher model.


### Run the frozen sender experiment

python -m an1_meaning_engine.experiment_frozen_sender

This will:

1. Load the frozen teacher  
2. Extract 64 dimensional headers  
3. Train the AN1 head  
4. Evaluate accuracy  
5. Compute latency and FLOPs  
6. Print the full benchmark summary  

All seeds are fixed. Everything is reproducible.

---

an1_meaning_engine/
    data.py
    teacher.py
    an1_head.py
    experiment_frozen_sender.py
    baselines/
        experiment_pixels.py
        experiment_scratch_block1.py

scripts/
    train_teacher.py

checkpoints/
results/
requirements.txt
LICENSE
PATENT_NOTICE.md
README.md

Entry point:  
python -m an1_meaning_engine.experiment_frozen_sender

---

## Notes

The teacher is always frozen

All teacher calls are wrapped in torch.no_grad()

No gradients flow into the teacher

Latency uses warmup, synchronization, and perf_counter

FLOP reduction is computed analytically from layer dimensions

No proprietary or symbolic internals are included


This is a clean public demonstration of a simple phenomenon.

---

## Dedication

Dedicated to my late father, Asad Shamim, whose loss opened the path that led me here.  
To my mother, Anni Shamim, whose living light carried me forward.
