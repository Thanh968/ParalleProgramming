# CSC14120 - Parallel Programming's Final Project

## Project introduction
- Requirement: Implement an Artificial Neural Network (ANN) utilizing CUDA for parallel computation capability to speed up traning process.
- Group members:
    | Student ID | Name |
    | :---: | --- |
    | 21120298 | Chiêm Bỉnh Nguyên |
    | 21120334 | Nguyễn Đình Thành |
    | 21120365 | Tô Hiển Vinh |

## Project structure
```
.
├── cpu
│   └── source.cu
├── cuda
│   ├── common.hpp
│   ├── v1.cu
│   ├── v2.cu
│   ├── v3.cu
│   └── v4.cu
├── data
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── val-images-idx3-ubyte
│   └── val-labels-idx1-ubyte
├── default_training_script.sh
├── README.md
├── split_dataset.py
├── train-images-idx3-ubyte
└── train-labels-idx1-ubyte
```
- `cpu`: sequential CPU code with no engineering optimization.
- `cuda`:
  - `v1.cu` is the naive interpretation of CPU version.
  - `v2.cu` utilizes CUDA streams for overlapping tasks (see the report/notebook for more details).
  - `v3.cu` optimizes matrix multiplication operation with 2D blocktiling.
  - `v4.cu` combines `v2` and `v3`.
- `split_dataset.py`: split the dataset into training set and validation set.

## Compiling source code
- The repository comes with an already-splitted dataset. If you want to re-split it:
  - Initialize a virtual environment (optional):
    ```bash
    $ python3 -m venv .venv
    ```
  - Run the script:
    ```bash
    $ python3 split_dataset.py --images-path <path-to-train-images> --labels-path <path-to-train-labels> --output-dir <output-directory path>
    ```
    For example:
    ```bash
    $ python3 split_dataset.py --images-path train-images-idx3-ubyte --labels-path train-labels-idx1-ubyte --output-dir data
    ```
- For `cuda`'s versions, compile with:
    ```bash
    $ nvcc cuda/v<x>.cu -o main
    ```
    For debugging, add the `DEBUG` definition:
    ```bash
    $ nvcc cuda/v<x>.cu -o main -DDEBUG
    ```
- Run executable:
    ```bash
    $ ./main --train-images <path-to-train-images> --train-labels <path-to-train-labels> --val-images <path-to-validation-images> --val-labels <path-to-validation-labels> --save-checkpoint <path-to-save-model-weights> --num-epochs <number-of-training-epochs>
    ```
    An example command can be found in `default_training_script.py`

## Demo video