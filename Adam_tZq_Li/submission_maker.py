import numpy as np
ADAM="True"
XJ1B="3j1b"
PARAMETER="Dropout_final_110_4"

def arraymaker(start, stop):
    first_entry = True
    for i in range(start, stop):
        if (first_entry == True):
            first_entry = False
            array = np.arange(10**(-i-1), 10**(-i), 10**(-i-1))
        else:
            array_temp = np.arange(10**(-i-1), 10**(-i), 10**(-i-1))
            array = np.concatenate([array, array_temp])
    return array

LR=np.arange(6.5e-5, 7e-5, 1e-5)
NODES=np.arange(110, 111, 5)
LAYERS=np.arange(4, 5, 1)
DROPOUT=np.arange(0.01, 0.99, 0.01)
VALIDATION=np.arange(0.2, 0.3, 0.1)
BATCHSIZE=np.arange(1024, 1025, 1)
DECAY=arraymaker=np.arange(1e-7, 2e-7, 1e-7)
MOMENTUM=np.arange(0.8, 0.9, 0.1)
EPOCH=np.arange(1000, 1001, 1)
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
                                    table=table + " " + str(b) + " " + str(round(de, 8)) + " " + str(round(m, 4)) + " " + str(e) + " " + XJ1B + " " + PARAMETER + "\n"

print(table)
