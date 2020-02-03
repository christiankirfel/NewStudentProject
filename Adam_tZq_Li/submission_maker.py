import numpy as np
ADAM="True"
XJ1B="2j1b"
"""
first_entry = True
for i in range(7):
    if (first_entry == True):
        first_entry = False
        LR=np.arange(10**(-i-1), 10**(-i), 10**(-i-1))
    else:
        LR_temp=np.arange(10**(-i-1), 10**(-i), 10**(-i-1))
        LR=np.concatenate([LR, LR_temp])
"""
LR=np.arange(1e-4, 2e-4, 1e-4)
NODES=np.arange(80, 81, 2)
LAYERS=np.arange(1, 25, 1)
DROPOUT=np.arange(0.10, 0.2, 0.1)
VALIDATION=np.arange(0.2, 0.3, 0.1)
BATCHSIZE=np.arange(1024, 1025, 1)
DECAY=np.arange(1e-8, 2e-8, 1e-8)
MOMENTUM=np.arange(0.2, 0.3, 0.1)
EPOCH=np.arange(400, 401, 2)
table=""

for lr in LR:
    for n in NODES:
        for l in LAYERS:
            for d in DROPOUT:
                for v in VALIDATION:
                    for b in BATCHSIZE:
                        for de in DECAY:
                            for m in MOMENTUM:
                                for e in EPOCH:
                                    table=table + ADAM + " " + str(round(lr, 9)) + " " + str(n) + " " + str(l) + " " + str(round(d, 2)) + " " + str(v)
                                    table=table + " " + str(b) + " " + str(round(de, 8)) + " " + str(round(m, 4)) + " " + str(e) + " " + XJ1B + "\n"

print(table)
