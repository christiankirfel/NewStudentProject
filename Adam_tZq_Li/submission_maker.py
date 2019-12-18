import numpy as np
ADAM="True"
XJ1B="3j1b"
LR=np.arange(0.00001, 0.001, 0.00001)
NODES=np.arange(80, 81, 1)
LAYERS=np.arange(7, 8, 1)
DROPOUT=np.arange(0.10, 0.3, 0.1)
VALIDATION=np.arange(0.8, 0.9, 0.1)
BATCHSIZE=np.arange(1024, 1025, 1)
DECAY=np.arange(1e-8, 2e-8, 1e-8)
MOMENTUM=np.arange(0.2, 0.3, 0.1)
EPOCH=np.arange(150, 151, 50)
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
                                    table=table + " " + str(b) + " " + str(round(de, 8)) + " " + str(round(m, 4)) + " " + str(e) + XJ1B + "\n"

print(table)
