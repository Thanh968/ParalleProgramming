#!/bin/bash

./main --train-images data/train-images-idx3-ubyte --train-labels data/train-labels-idx1-ubyte --val-images data/val-images-idx3-ubyte --val-labels data/val-labels-idx1-ubyte --save-checkpoint checkpoint.weight --num-epochs 1

# ./main --train-images train-images-idx3-ubyte --train-labels train-labels-idx1-ubyte --val-images train-images-idx3-ubyte --val-labels train-labels-idx1-ubyte --save-checkpoint checkpoint.weight --num-epochs 1