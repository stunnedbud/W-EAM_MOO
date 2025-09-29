import numpy as np
from scipy.stats import ranksums

x = [20, 22, 19, 20, 22, 18, 24, 20, 19, 24, 26, 13]
y = [38, 37, 33, 29, 14, 12, 20, 22, 17, 25, 26, 16]
print(ranksums(x, y))


def F1score(pre, rec):
    if (pre + rec) > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.0


# MNIST
base_F1 = [0.961, 0.970, 0.956, 0.945, 0.939, 0.928, 0.908, 0.886]
base_pre = [0.992, 0.984, 0.995, 0.998, 0.999, 0.999, 0.999, 1.00]
base_rec = [0.932, 0.956, 0.919, 0.896, 0.886, 0.867, 0.832, 0.796]

smac_pre = [
    0.997182237,
    0.999427725,
    0.999716268,
    0.999431025,
    0.999288946,
    0.99972106,
    0.999432273,
    0.999131774,
    0.999584105,
    1,
]
smac_rec = [0.905665367, 1, 1, 1, 1, 1, 1, 1, 1, 1]
smac_F1 = []
for i in range(len(smac_pre)):
    smac_F1.append(F1score(smac_pre[i], smac_rec[i]))

print("smac F1s")
print(smac_F1)

smsemoa_F1 = [
    0.960669256,
    0.901674965,
    0.924931824,
    0.919837401,
    0.885384749,
    0.919092558,
    0.922592098,
    0.923204094,
    0.928966137,
    0.922748276,
]
smsemoa_pre = [
    0.980433242,
    0.953395697,
    0.96242654,
    0.977723581,
    0.918927135,
    0.957586434,
    0.972054206,
    0.991204418,
    0.988621298,
    0.962630002,
]
smsemoa_rec = [
    0.941979912,
    0.860826276,
    0.890820398,
    0.869011963,
    0.863645922,
    0.885147655,
    0.880629853,
    0.864253554,
    0.877151584,
    0.888182282,
]


print("smac F1 vs base")
print(ranksums(smac_F1, base_F1))
print("smac Precision vs base")
print(ranksums(smac_pre, base_pre))
print("smac Recall vs base")
print(ranksums(smac_rec, base_rec))

print("smsemoa F1 vs base")
print(ranksums(smsemoa_F1, base_F1))
print("smsemoa Precision vs base")
print(ranksums(smsemoa_pre, base_pre))
print("smsemoa Recall vs base")
print(ranksums(smsemoa_rec, base_rec))
