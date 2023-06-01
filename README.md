# The accompanying code for LEAP: Inductive Link Prediction via Learnable Topology Augmentation
## Requirements!

-   Python 3.6+
-   PyTorch 1.10+
-   PyG 2.0+
-   Numpy 1.17.2+
-   Networkx 2.3+
-   SciPy 1.5.4+


### Example usage

### For homogeneous graphs experiment

```sh
$ python leap.py
```

### For heterogeneous graphs experiment

```sh
$ python leap_Hetero.py
```
### Options

`--transductive:`
The type of the link prediction task: transductive or Inductive. Default is `False`

`--device:` The device to run on. Default is `cuda:0`

`--name`: The name of the dataset to run on. Default is Wikipedia

`--epochs`: The number of epochs. Default is 10000
