##Testing Evolution
import pandas 
import numpy as np

anzahl=100

lr=np.random.rand(anzahl, 3)
for i in range(100):
    lr=np.sort(lr, axis=0)
    lr=lr[0:(int(anzahl/2)), :]
    lr_2=np.random.rand(int(anzahl/2), 3)
    lr=np.concatenate([lr, lr_2])
    
print(lr)