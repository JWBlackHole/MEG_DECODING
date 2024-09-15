# Torch Dataset

## how to load data only when accessed?

need a class getData that inherit torch.utils.data.Dataset

need 2 method: `__getitem__` and `__len__`


DataLoader --> calls __getitem__ for each batch during the training
__getitem__ define what should each batch gives