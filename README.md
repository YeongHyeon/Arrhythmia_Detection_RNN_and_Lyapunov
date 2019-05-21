Arrhythmia Detection in ECG based on RNN Encoder–Decoder with Lyapunov Exponent
=====

## Introduction
This repository provides the source code of the paper <a href="https://onlinelibrary.wiley.com/doi/full/10.1002/tee.22927">"Arrhythmia Detection in Electrocardiogram based on Recurrent Neural Network Encoder–Decoder with Lyapunov Exponent"</a>.

<div align="center">
  <img src="./figures/model.png" width="300">  
  <p>The structure of RNN Encoder-Decoder.</p>
</div>

<div align="center">
  <img src="./figures/example.png" width="300">  
  <p>The example of arrhythmia detection. Label '100', '103', and '107' are normal (healthy), mixed (normal and arrhythmia), and arrhythmia respectively.</p>
</div>

## Requirements
* Python 3.5.2  
* Tensorflow 1.4.0  
* Numpy 1.13.3  
* Scipy 1.2.0  
* Matplotlib 3.0.2  

## Database
You can download the original <a href="https://physionet.org/physiobank/database/mitdb/">MIT-BIH Arrhythmia Database</a> in physiobank.


### BibTeX
```
@Article{doi:10.1002/tee.22927,
  author = {Park, YeongHyeon and Yun, Il Dong},
  title = {Arrhythmia detection in electrocardiogram based on recurrent neural network encoder–decoder with Lyapunov exponent},
  journal = {IEEJ Transactions on Electrical and Electronic Engineering},
  volume = {0},
  number = {0},
  keywords = {arrhythmia detection, recurrent neural network encoder-decoder Lyapunov exponent},
  doi = {10.1002/tee.22927},
  url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/tee.22927},
  eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/tee.22927},
}
```
