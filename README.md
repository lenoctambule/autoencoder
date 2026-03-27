# Python AutoEncoder from scratch using Numpy

## Usage 

1. Install requirements :
```sh
$ pip install -r requirements.txt
```

2. Optionally run mnist_test.py.
```sh
$ py mnist_test.py 
```

## Training

Instatiate an `Autoencoder` object :
```py
from autoencoder import Autoencoder

autoencoder = Autoencoder(in_len=300, bottleneck=50, 0.001, relu)
```
And then via the `train_dataset` method to train over a dataset :
```py
autoencoder.train_dataset(data)
```
Or via the `train` to input each data points iteratively :
```py
autoencoder.train(v)
```

## Inference

Use your `Autoencoder` object with the `encode` and `decode` methods like so :
```py
example = ...
code = autoencoder.encode(example)
output = autoencoder.decode(code)
```
