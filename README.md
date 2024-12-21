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
├── cuda
│   ├── common.hpp
│   ├── v1.cu
│   ├── v2.cu
│   ├── v3.cu
│   ├── v4.cu
│   └── v5.cu
├── data
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── val-images-idx3-ubyte
│   ├── val-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── default_training_script.sh
├── weight.txt
├── cpu.cu
└── README.md
```
- `cpu`: sequential CPU code with no engineering optimization.
- `cuda`:
  - `v1.cu` is the naive interpretation of CPU version.
  - `v2.cu` utilizes CUDA streams for overlapping tasks (see the report/notebook for more details).
  - `v3.cu` optimizes matrix multiplication operation using shared memory.
  - `v4.cu` optimizes matrix multiplication operation with 2D blocktiling.
  - `v5.cu` combines `v2` and `v4`.

## Compiling source code
- For `cpu` version, compile with:
    ```bash
    $ nvcc cpu.cu -o cpu.out
    ```
    Run cpu version:
    ```bash
    $ ./cpu.out
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
    ````

## Demo video