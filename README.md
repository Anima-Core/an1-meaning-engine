# AN1 Meaning Engine
64 Dimensional Frozen Sender

---

This repository contains a minimal PyTorch implementation of the AN1 Meaning Engine experiment on CIFAR-10.  
It explores what happens when meaning is captured before the matrix rather than after it.

By freezing a ResNet18 and exposing only a 64 dimensional intention header from an early layer, the AN1 head learns to reconstruct the teacher’s behavior with remarkable efficiency.

---

## Why This Experiment Matters


Most acceleration work focuses on kernels and low level optimization.  
This experiment shifts the learning problem itself.

The heavy ResNet18 stays completely frozen.  
AN1 only processes a 64 dimensional header.

The result is an unexpectedly large separation between compute cost and predictive power.

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

This is a simple experiment, yet it reveals something important.  
Not all intelligence needs the full stack.  
This experiment suggests that intelligence may not require full computation, only the right early signal.

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

## Repository Structure

an1_meaning_engine/  
    data.py  
    teacher.py  
    an1_head.py  
    experiment_frozen_sender.py  
    metrics.py  

scripts/  
    train_teacher.py  

checkpoints/  
requirements.txt  
LICENSE  
PATENT_NOTICE.md  
README.md  

Entry point:  
python -m an1_meaning_engine.experiment_frozen_sender

---

## Notes

The teacher is always frozen.  
All teacher calls are inside torch.no_grad().  
No gradients enter the teacher.

Latency metrics use warmup, synchronization, and perf_counter.  
FLOP reduction is computed directly from MLP dimensions.

No symbolic or proprietary internals are included.  
This is a clean, public-safe research demonstration.

---

## Dedication

Dedicated to my late father, Asad Shamim, whose loss opened the path that led me here.  
To my mother, Anni Shamim, whose living light carried me forward.