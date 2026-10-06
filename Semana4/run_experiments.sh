#!/bin/bash

# Experimento 1: Baseline defaults
echo ">>> Ejecutando Experimento 1 (Default)..."
python3 src/train.py

# Experimento 2: Strong regularization
echo ">>> Ejecutando Experimento 2 (C=0.01)..."
python3 src/train.py --c_param 0.01

# Experimento 3: Weak regularization
echo ">>> Ejecutando Experimento 3 (C=10.0)..."
python3 src/train.py --c_param 10.0

# Experimento 4: Class weight balanced
echo ">>> Ejecutando Experimento 4 (class_weight=balanced)..."
python3 src/train.py --class_weight balanced

# Experimento 5: Class weight balanced + Strong regularization
echo ">>> Ejecutando Experimento 5 (class_weight=balanced, C=0.1)..."
python3 src/train.py --class_weight balanced --c_param 0.1
