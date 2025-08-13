from pytorch_ecc import Data


data = Data.load("..")

data.record_n(10, 0.0003, True, autosave=5)
