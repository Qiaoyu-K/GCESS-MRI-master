## [MRI Reconstruction with Enhanced Self-Similarity Using Graph Convolutional Network](https://www.researchsquare.com/article/rs-2702846/v1) ()

---

by Qiaoyu Ma, Zongying Lai, Zi Wang, Yiran Qiu, Biao Qu, Haotian Zhang, Xiaobo Qu Jimei 
University&Xiamen University

### Citation

@article{ma2023mri,
  title={MRI Reconstruction with Enhanced Self-Similarity Using Graph Convolutional Network},
  author={Ma, Qiaoyu and Lai, Zongying and Wang, Zi and Qiu, Yiran and Qu, Biao and Zhang, Haotian and Qu, Xiaobo},
  year={2023}
}

### Dependencies and Installation

* python3
* PyTorch>=1.8
* NVIDIA GPU+CUDA
* numpy
* matplotlib

### Datasets Preparation

Dataset are available at: https://pan.baidu.com/s/1Zj9V2GOD882cFTXMLbMDyQ?pwd=ngjf
Dataset Source: [[Learning a Variational Network for Reconstruction of Accelerated MRI Data](https://arxiv.org/abs/1704.00447)]

<details>
<summary> FILE STRUCTURE </summary>

```
    FFA-Net
    |-- 01_Graph_save_gen.py
    |-- 02_train.py
    |-- 03_test.py 
    |-- README.md
    |-- model
        |-- model_GCESS.py
    |-- pyds_lib
        |-- utils.pyd
    |-- save_models
    |-- Tool
        |-- __init__.py
        |-- evaluation.py
        |-- utils_network.py
           
```

</details>

