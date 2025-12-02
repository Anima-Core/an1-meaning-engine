#!/bin/bash

# Set PYTHONPATH to repo root
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd):$PYTHONPATH"

# Run the experiment
python -m an1_meaning_engine.experiment_frozen_sender

