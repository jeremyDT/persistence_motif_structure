
import pandas as pd
import numpy as np
import random
import itertools
import networkx as nx


temp = [y for z in store_persistent_tri for x in z[fit_motif:] for y in x]
#temp = [y for x in store_persistent_tri[0][fit_motif:] for y in x]

print('created temp')
found = []
dict = {}
counter = 0
for j in temp:
    counter +=1
    if counter % 10000 == 0:
        print(counter)
    if j in found:
        dict[found.index(j)][0] = dict[found.index(j)][0] + 1
        #dict[str(j)][0] = dict[str(j)][0] + 1
    else:
        found.append(j)
        dict[found.index(j)] = [1, j]
print(len(found))
counts = []
for i in dict.keys():
    counts.append(dict[i][0])

plt.hist(counts)
plt.show()

def make_clique_matrix(clique_list, N):

    matrix = np.zeros(shape=(N, N))

    for h in clique_list:

        for k in itertools.permutations(h, 2):
            matrix[k[0], k[1]] +=1

    return matrix

tri_matrix = make_clique_matrix(store_persistent_tri, 100)

G = nx.from_numpy_matrix(tri_matrix)

nx.draw(G, with_labels=True, font_weight='bold')

strong_tr = []
threshold = 100000 #( time_length  - fit_motif )*iterations *0.70
for i in dict.keys():
    if dict[i][0] > threshold:
        print(i, dict[i])
        strong_tr.append(dict[i][1])

threshold = 10000 #( time_length  - fit_motif )*iterations *0.60

mid_tr = []
for i in dict.keys():
    if dict[i][0] > threshold:
        print(i, dict[i])
        #mid_tr.append(i)
        mid_tr.append(dict[i][1])

print('triangles in strong', len(strong_tr))
df_names = pd.read_csv('./trial_matrix_names.csv', header = 0, parse_dates=True, index_col=0)
df= pd.read_excel('./data_for_mkt_comparison/MORE_stocks.xlsx', header = 0, index_col=0, parse_dates=True)

df = df[list(df_names.columns)]

names = []
for i in strong_tr:
    for temp_hedge in i:

        #temp_hedge = random.sample(i, 1)[0]
        temp_name = df.columns[temp_hedge]
        if temp_name not in names:
            names.append(temp_name)

df_filter = df.drop(names, axis = 1)
print(df.shape, df_filter.shape)

################################################

names = []
for i in mid_tr:
    #i = strong_tr[1]
    for temp_hedge in random.sample(i, 1):
        # temp_hedge = random.sample(i, 1)[0]
        temp_name = df.columns[temp_hedge]
        if temp_name not in names:
            names.append(temp_name)

df_filter = df_names[names + random.sample(list(df_names.columns), 10)]

sigmas = []
N = len(names)
weights = np.full(N, 1/N)

cov_ret = df_filter.cov()

cov_ret = cov_ret.values
sigma = 0

for k in range(N):
    for m in range(N):
        if k != m:
            sigma += (cov_ret[k,m])

sigmas.append(sigma)

weights = np.full(N, 1/N)
running_sigma = 0
for f in range(100):

    df_random = df_names[random.sample(list(df_names.columns), N) + random.sample(list(df_names.columns), 10)]

    cov_ret = df_random.cov()

    cov_ret = cov_ret.values
    sigma = 0

    for k in range(N):
        for m in range(N):
            if k != m:
                sigma += (cov_ret[k,m])
    running_sigma += sigma

sigmas.append(running_sigma/100)

print(sigmas)





'''for k in range(100):

    weights = np.ones(100)

    names = []
    for i in strong_tr:
        temp_hedge = random.sample(i, 1)[0]
        temp_name = df.columns[temp_hedge]
        if temp_name not in names:
            names.append(temp_name)
            weights[temp_hedge] *= -1

    cov_ret = df_ret.cov()

    cov_ret = cov_ret.values
    sigma = 0

    for k in range(100):
        for m in range(100):
            if k != m:
                sigma += (cov_ret[k,m])
    print(sigma)
    temp_sigmas = []
    for h in range(100):
        weights_rand = np.ones(100)

        names_rand = []
        while len(names_rand) < len(names):
            temp_hedge = np.random.randint(0, 100)
            temp_name = df.columns[temp_hedge]
            if temp_name not in names_rand:
                names_rand.append(temp_name)
                weights_rand[temp_hedge] *= -1

        sigma = 0

        for k in range(100):
            for m in range(100):
                if k != m:
                    sigma += (weights_rand[k] * weights_rand[m] * cov_ret[k,m])
        #print(sigma)

        temp_sigmas.append(sigma)

    sigmas[0].append(temp_sigmas)

    sigma = 0

    for k in range(100):
        for m in range(100):
            if k != m:
                sigma += (weights[k] * weights[m] * cov_ret[k,m])
    #print(sigma)
    sigmas[1].append(sigma)




    weights = np.ones(100)
    names = []
    for i in mid_tr:
        temp_hedge = random.sample(i, 1)[0]
        temp_name = df.columns[temp_hedge]
        if temp_name not in names:
            names.append(temp_name)
            weights[temp_hedge] *= -1

    names_rand = []
    while len(names_rand) < len(names):
        temp_hedge = np.random.randint(0, 100)
        temp_name = df.columns[temp_hedge]
        if temp_name not in names_rand:
            names_rand.append(temp_name)
            weights_rand[temp_hedge] *= -1


    cov_ret = df_ret.cov()

    cov_ret = cov_ret.values
    sigma = 0

    for k in range(100):
        for m in range(100):
            if k != m:
                sigma += (cov_ret[k,m])
    #print(sigma)
    temp_sigmas = []
    for h in range(100):

        weights_rand = np.ones(100)

        names_rand = []
        while len(names_rand) < len(names):
            temp_hedge = np.random.randint(0, 100)
            temp_name = df.columns[temp_hedge]
            if temp_name not in names_rand:
                names_rand.append(temp_name)
                weights_rand[temp_hedge] *= -1

        sigma = 0

        for k in range(100):
            for m in range(100):
                if k != m:
                    sigma += (weights_rand[k] * weights_rand[m] * cov_ret[k, m])
        #print(sigma)

        temp_sigmas.append(sigma)

    sigmas[2].append(temp_sigmas)

    sigma = 0

    for k in range(100):
        for m in range(100):
            if k != m:
                sigma += (weights[k] * weights[m] * cov_ret[k,m])
    #print(sigma)
    sigmas[3].append(sigma)
print(sigmas[1][0], sigmas[0][0])'''


'''from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Read in price data
#df = pd.read_csv("tests/stock_prices.csv", parse_dates=True, index_col="date")

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(df[df.index[1100]:df.index[1200]])
S = risk_models.sample_cov(df[df.index[1100]:df.index[1200]])
print(df[df.index[1100]:df.index[1200]].shape)
# Optimise for maximal Sharpe ratio
ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.1))
weights = ef.efficient_return(target_return=0.8)#max_sharpe()
print(ef.portfolio_performance(verbose=True))

mu = expected_returns.mean_historical_return(df_filter[df.index[1100]:df.index[1200]])
S = risk_models.sample_cov(df_filter[df.index[1100]:df.index[1200]])

print(df_filter[df.index[1100]:df.index[1200]].shape)

# Optimise for maximal Sharpe ratio
ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.1))
weights = ef.efficient_return(target_return=0.8)
print(ef.portfolio_performance(verbose=True))'''


#df_ret = df / df.shift(1)
#df_ret.drop(0, axis = 0, inplace = True)
#df_ret = df_ret.iloc[126+fit_motif:600]

'''sigmas = [[], [], [], []]
for k in range(100):

    weights = np.ones(100)


    import random
    names = []
    for i in strong_tr:
        temp_hedge = random.sample(i, 1)[0]
        temp_name = df.columns[temp_hedge]
        if temp_name not in names:
            names.append(temp_name)
            weights[temp_hedge] *= -1

    cov_ret = df_ret.cov()

    cov_ret = cov_ret.values
    sigma = 0

    for k in range(100):
        for m in range(100):
            if k != m:
                sigma += (cov_ret[k,m])
    print(sigma)
    temp_sigmas = []
    for h in range(100):
        weights_rand = np.ones(100)

        names_rand = []
        while len(names_rand) < len(names):
            temp_hedge = np.random.randint(0, 100)
            temp_name = df.columns[temp_hedge]
            if temp_name not in names_rand:
                names_rand.append(temp_name)
                weights_rand[temp_hedge] *= -1

        sigma = 0

        for k in range(100):
            for m in range(100):
                if k != m:
                    sigma += (weights_rand[k] * weights_rand[m] * cov_ret[k,m])
        #print(sigma)

        temp_sigmas.append(sigma)

    sigmas[0].append(temp_sigmas)

    sigma = 0

    for k in range(100):
        for m in range(100):
            if k != m:
                sigma += (weights[k] * weights[m] * cov_ret[k,m])
    #print(sigma)
    sigmas[1].append(sigma)




    weights = np.ones(100)
    names = []
    for i in mid_tr:
        temp_hedge = random.sample(i, 1)[0]
        temp_name = df.columns[temp_hedge]
        if temp_name not in names:
            names.append(temp_name)
            weights[temp_hedge] *= -1

    names_rand = []
    while len(names_rand) < len(names):
        temp_hedge = np.random.randint(0, 100)
        temp_name = df.columns[temp_hedge]
        if temp_name not in names_rand:
            names_rand.append(temp_name)
            weights_rand[temp_hedge] *= -1


    cov_ret = df_ret.cov()

    cov_ret = cov_ret.values
    sigma = 0

    for k in range(100):
        for m in range(100):
            if k != m:
                sigma += (cov_ret[k,m])
    #print(sigma)
    temp_sigmas = []
    for h in range(100):

        weights_rand = np.ones(100)

        names_rand = []
        while len(names_rand) < len(names):
            temp_hedge = np.random.randint(0, 100)
            temp_name = df.columns[temp_hedge]
            if temp_name not in names_rand:
                names_rand.append(temp_name)
                weights_rand[temp_hedge] *= -1

        sigma = 0

        for k in range(100):
            for m in range(100):
                if k != m:
                    sigma += (weights_rand[k] * weights_rand[m] * cov_ret[k, m])
        #print(sigma)

        temp_sigmas.append(sigma)

    sigmas[2].append(temp_sigmas)

    sigma = 0

    for k in range(100):
        for m in range(100):
            if k != m:
                sigma += (weights[k] * weights[m] * cov_ret[k,m])
    #print(sigma)
    sigmas[3].append(sigma)
print(sigmas[1][0], sigmas[0][0])'''