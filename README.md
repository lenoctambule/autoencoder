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
from activations import LeakyReLU

autoencoder = Autoencoder(
    [768, 64, 16],
    [16, 64, 768],
    0.01,
    LeakyReLU()
)
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

Use your `Autoencoder` object with the `encode`, `decode`, `forward` methods like so :
```py
example = ...
code = autoencoder.encode(example)
output = autoencoder.decode(code)
output, code = autoencoder.forward(example)
```
