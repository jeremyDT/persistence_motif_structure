import itertools
import numpy as np
import matplotlib.pyplot as plt

all_tri = list(itertools.combinations_with_replacement(range(100), 3))

def tr_strength(triangle, W):

    strength = 0

    for edge in itertools.combinations(triangle, 2):

        strength += W[edge[0], edge[1]]
    return strength

strong_tr = []
threshold = 23000 #( time_length  - fit_motif )*iterations *0.70
for i in dict.keys():
    if dict[i][0] > threshold:
        print(i, dict[i])
        strong_tr.append(dict[i][1])
probabs = []
for file in file_list:

    print(file)

    W = np.loadtxt(open(file, "rb"), delimiter=",")

    strengths = [tr_strength(x, W) for x in all_tri]

    sorted_tri = [set(x) for _, x in sorted(zip(strengths, all_tri), reverse=True)][:10]

    matching = [x for x in sorted_tri if x in strong_tr]

    probabs.append(len(matching))

plt.hist(probabs, bins = 50)
plt.show()