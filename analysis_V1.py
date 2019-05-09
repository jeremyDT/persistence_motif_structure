import numpy as np
import pandas as pd
import os
from TMFG_square_with_method import TMFG
import matplotlib.pyplot as plt
import math
from joblib import Parallel, delayed
import multiprocessing
import datetime
import seaborn as sns
from itertools import combinations, permutations
from sklearn.metrics import r2_score, mean_squared_error
import networkx as nx
from scipy import optimize
import glob

def load_elaborated(RANDOM = True, n = 100, exchange_names = ['NYSE', 'germany', 'italy', 'israel'], buckets = 1):


    #exchange_names = ['NYSE', 'germany', 'italy', 'israel', 'shenzen']#['NYSE', 'germany', 'italy', 'greece', 'UK', 'turkey', 'israel', 'saudi', 'mexico', 'bombay', 'euronext', 'canada', 'russia', 'brasil'] #'australia'
    #pair_names = ['EURUSD']

    exchanges = {}
    #np.arange(0, 24, 2), np.arange(1, 24, 2)

    #col_indices = np.insert(col_indices, 0, 0, axis=0)
    if RANDOM:
        for name in exchange_names:
            exchanges[name] = {}
            to_read = './data_for_mkt_comparison/' + name + '.csv'
            temp = pd.read_csv(to_read, header=0, index_col=0, parse_dates=True)
            for bucket in range(buckets):

                exchanges[name][bucket] = temp.sample(n= n, replace=False, axis = 1) #random_state=1

                print(exchanges[name][bucket].shape)

                exchanges[name][bucket].to_csv('trial_matrix_names.csv')


    '''else:
        for name in exchange_names:
            to_read = './data_for_mkt_comparison/' + name + '.csv'
            exchanges[name] = pd.read_csv( to_read,  header = 0, index_col=0, parse_dates=True)

            print(name, exchanges[name].shape)
            #print(exchanges[name])

    return exchanges'''




def graph_layers(data, T, dt, roll_step, theta):
    w = kendall_exp_weights(dt, theta)
    w = w.reshape(len(w), 1)
    correl_matrices = []
    for i in range(dt, T + 1, roll_step):
        Y = data[(i - dt):i, :]

        correl_matrices.append(kendalltau(Y, w))
    return correl_matrices


def edge_filtered(edge_w, apply_TMFG = True):

    edge_w_filtered = []

    num_cores = multiprocessing.cpu_count() - 1

    edge_w_filtered = Parallel(n_jobs=num_cores)(delayed(TMFG_A)(i) for i in edge_w)

    return np.dstack(edge_w_filtered)


def TMFG_A(W):
    #print(counter_glob)
    #print(W)
    A, tri, separators, cliques, cliqueTree = TMFG(W, method='sq')
    return np.asarray(A)

def make_3_tensor(filtered_matrices):
    T = filtered_matrices.shape[2]
    N = filtered_matrices.shape[0]

    list_all_times = []

    for t in range(T):
        matrix_t = filtered_matrices[:, :, t]

        temp = [set((i, j, k)) for i in range(N - 1) for j in range(i + 1, N) for k in range(j + 1, N) if (bool(matrix_t[i, j]) and bool(matrix_t[j,k]) and bool(matrix_t[i, k]))]

        list_all_times.append(temp)

    return list_all_times

def make_3_matrix(triangle_list, N):

    matrix = np.zeros(shape=(N, N))

    for h in triangle_list:
        h = list(h)
        matrix[h[0], h[1]] = 1
        matrix[h[1], h[0]] = 1
        matrix[h[0], h[2]] = 1
        matrix[h[2], h[0]] = 1
        matrix[h[2], h[1]] = 1
        matrix[h[1], h[2]] = 1
    return matrix

def produce_nets(data):

    temp = data.values
    T = temp.shape[0]


    dt = 126
    theta = 46
    roll_step = 1
    correl_matrices = graph_layers(temp, T, dt, roll_step, theta)
    edge_w_filtered = np.abs(edge_filtered(correl_matrices, apply_TMFG = True))

    '''edge_w_filtered = np.abs(
        np.load(os.path.join('./../correlation-matrices', timestamp, (matrix[:-4] + i))))
    edge_w_filtered = np.heaviside(edge_w_filtered, 0).astype(int)'''

    set_list = make_3_tensor(edge_w_filtered)

    return edge_w_filtered, set_list, np.dstack(correl_matrices)


def produce_persist(edge_w_filtered, set_list, iterations = 200, time_length = 600):

    T = edge_w_filtered.shape[2]
    N = edge_w_filtered[0].shape[0]

    memory_array = np.zeros(time_length)
    edge_memory_array = np.zeros(time_length)
    hard_array = np.zeros(time_length)
    hard_edge_array = np.zeros(time_length)
    memory_array_chain = np.zeros(time_length)
    hard_array_chain = np.zeros(time_length)

    store_triangulated = np.zeros(shape=(N, N, time_length))
    store_sets = []

    for u in range(0, iterations):

        temp_hard = set_list[u]
        temp_edge_hard = edge_w_filtered[:, :, u]
        print('MOTIFS', len(set_list[u]))
        #print('N: {} T: {}'.format(N, T))
        for g in range(u, time_length + u):

            persistent = [set(m) for m in set_list[u] if
                          m in set_list[g]]  # np.minimum(persistent, edge_w_filtered[:, :, g])

            temp_tr = make_3_matrix(persistent, N)

            store_sets.append(persistent)

            store_triangulated[:, :, g - u] += temp_tr

            edge_persistent = np.minimum(edge_w_filtered[:, :, u], edge_w_filtered[:, :, g])

            memory_array[g - u] += len(persistent) / len(set_list[u])
            edge_memory_array[g - u] += np.sum(edge_persistent) / np.sum(edge_w_filtered[:, :, u])

    store_triangulated /= iterations

    #xdata = np.arange(1, len(memory_array) + 1)

    y_edge =  np.asarray(edge_memory_array / iterations)
    y_triangle = np.asarray((memory_array / iterations))




    return y_edge, y_triangle, store_triangulated, store_sets

def opt_pl(logx, logy, xdata, R2 = True):

    powerlaw = lambda x, amp, index: amp * (x ** index)

    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y: (y - fitfunc(p, x))

    pinit = [1.0, -1.0]
    out = optimize.leastsq(errfunc, pinit,
                           args=(logx, logy), full_output=1)

    pfinal = out[0]
    covar = out[1]
    #print(pfinal)
    #print(covar)

    index = pfinal[1]
    amp = np.exp( pfinal[0] )

    indexErr = np.sqrt(covar[1][1])
    ampErr = np.sqrt(covar[0][0]) * amp

    if R2:
        #print(mean_squared_error(logy, fitfunc(pfinal, logx)))
        return mean_squared_error(logy, fitfunc(pfinal, logx)), [amp, index, ampErr, indexErr]



def opt_R2(ys, labels, name):

    break_points = []

    for y_temp, label in zip(ys, labels):


        ##########
        # Fitting the data -- Least Squares Method
        ##########
        # Define function for calculating a power law
        powerlaw = lambda x, amp, index: amp * (x ** index)

        # define our (line) fitting function
        fitfunc = lambda p, x: p[0] + p[1] * x
        errfunc = lambda p, x, y: (y - fitfunc(p, x))

        # Power-law fitting is best done by first converting
        # to a linear equation and then fitting to a straight line.
        # Note that the `logyerr` term here is ignoring a constant prefactor.
        #
        #  y = a * x^b
        #  log(y) = log(a) + b*log(x)
        #
        xdata = np.arange(1, len(y_temp) + 1)
        logx = np.log(xdata)
        logy = np.log(y_temp)
        #logy_tr = np.log(y_triangle)
        # logyerr = yerr / y_triangle
        r2s = np.zeros(2) + 1000
        params = [[], []]
        break_point = 0

        for i in range(10, xdata[-1] - 10):


            temp_edge, data_edge = opt_pl(logx[:i], logy[:i], xdata[:i], R2 = True)
            temp_tr, data_tr = opt_pl(logx[i:], logy[i:], xdata[i:], R2=True)
            #print(temp_edge, temp_tr, r2s)
            #temp_edge /= len(logx[:i])
            #temp_tr /= len(logx[i:])
            '''plt.scatter(i, temp_edge + temp_tr, label = '1')
            plt.scatter(i, temp_edge, label='2')
            plt.scatter(i, temp_tr, label='3')'''
            if (temp_edge + temp_tr) < np.sum(r2s):
                r2s[0] = temp_edge
                r2s[1] = temp_tr
                params[0] = data_edge
                params[1] = data_tr
                break_point = i
        #plt.show()

        break_points.append(break_point)

        print(break_point, r2s, params)

        amp, index, ampErr, indexErr = params[0]

        ##########
        # Plotting data
        ##########

        print('exponent edge 1  ', index)
        print(amp)

        plt.loglog(xdata, powerlaw(xdata, amp, index), label= label + ' 1')
        plt.scatter(xdata[:break_point], y_temp[:break_point])  # Data

        ##########
        # Plotting data
        ##########
        amp, index, ampErr, indexErr = params[1]

        print('exponent edge 2  ', index)
        print(amp)

        plt.loglog(xdata, powerlaw(xdata, amp, index), label= label + ' 2')
        plt.scatter(xdata[break_point:], y_temp[break_point:])  # Data





    #plt.text(5, 6.5, 'Ampli = %5.2f +/- %5.2f' % (amp, ampErr))
    #plt.text(5, 5.5, 'Index = %5.2f +/- %5.2f' % (index, indexErr))
    plt.title(name)

    plt.xlabel('X (log scale)')
    plt.ylabel('Y (log scale)')
    plt.xlim(1.0, len(xdata) + 1)
    plt.ylim(1, 1000)
    plt.legend()
    plt.savefig('for_abstract.eps', format='eps', dpi=1000)
    plt.show()

    return break_points

def core_consistency():
    iterations = 200

    time_length = 500

    buckets = 1

    n = 100

    data = load_elaborated(RANDOM = True, n = n, exchange_names = ['NYSE'], buckets = buckets)
    print('loaded')
    for key in data.keys():
        store_y_edge = np.zeros(time_length)
        store_y_triangle = np.zeros(time_length)
        tr_tensor = []

        dt_values = []
        break_edges = []
        break_motifs = []
        global dt
        global theta
        #dt = 126
        theta = 46

        plat_motifs = []

        for dt in [126]:#, 126, 252]:


            for bucket in data[key].keys():

                edge_w_filtered, set_list, correl_matrices = produce_nets(data[key][bucket])
                print('motifs')

                y_edge, y_triangle, tr_temp = produce_persist(edge_w_filtered, set_list, iterations = iterations, time_length = time_length)
                print(tr_temp)
                #tr_tensor.append(tr_temp)
                print('data')
                store_y_edge += y_edge
                store_y_triangle += y_triangle



            store_y_edge /= len(data[key].keys())
            store_y_triangle /= len(data[key].keys())

            fit_edge, fit_motif = opt_R2(store_y_edge, store_y_triangle, 'NYSE')

            tr_tensor = tr_temp[0][:, :, fit_motif:]
            tr_tensor = np.sum(tr_tensor, axis=2)

            plat_motifs.append(tr_tensor)
        from scipy import stats
        for k in plat_motifs[1:]:
            tau, p_value = stats.kendalltau(plat_motifs[0].flatten(), k.flatten())
            print('kendall tau {} p_value {}'.format(tau, p_value))
            tau, p_value = stats.pearsonr(plat_motifs[0].flatten(), k.flatten())
            print('pearson tau {} p_value {}'.format(tau, p_value))




if __name__ == "__main__":

    iterations = 200

    time_length = 900

    file_list = ['./matrices/{}.csv'.format(x) for x in range(1, 1132)]#glob.glob('./matrices/*.csv')

    print(len(file_list))

    W_init = np.loadtxt(open(file_list[0], "rb"), delimiter=",")

    print(W_init.shape)

    A_init, tri_init, separators_init, cliques_init, cliqueTree_init = TMFG(W_init, method='sq')

    tri_init = [set((tri_init[index, 0], tri_init[index, 1], tri_init[index, 2])) for index in range(tri_init.shape[0])]


    store_tri = []
    store_clique = []
    store_separator = []
    for file in file_list:

        print(file)

        W = np.loadtxt(open(file, "rb"), delimiter=",")

        A, tri, separators, cliques, cliqueTree = TMFG(W, method='sq')

        tri = [set((tri[index, 0], tri[index, 1], tri[index, 2])) for index in
               range(tri.shape[0])]

        separators = [set((separators[index, 0], separators[index, 1], separators[index, 2])) for index in
               range(separators.shape[0])]

        cliques = [set((cliques[index, 0], cliques[index, 1], cliques[index, 2], cliques[index, 3])) for index in
               range(cliques.shape[0])]

        store_tri.append(tri)

        store_clique.append(cliques)

        store_separator.append(separators)

    print(len(separators))

    motif_decay = np.zeros((time_length, iterations))
    separator_decay = np.zeros((time_length, iterations))
    clique_decay = np.zeros((time_length, iterations))

    store_persistent_tri = []

    store_persistent_tetrahedron = []

    store_persistent_sep = []


    for u in range(iterations):

        print(u)

        tri_init = store_tri[u]
        separator_init = store_separator[u]
        clique_init = store_clique[u]
        store_persistent_tri.append([])
        store_persistent_tetrahedron.append([])
        store_persistent_sep.append([])
        for g in range(u, u + time_length):


            tri = store_tri[g]

            clique = store_clique[g]

            separators = store_separator[g]


            persistent_tri = [m for m in tri_init if
                      m in tri]

            store_persistent_tri[-1].append(persistent_tri)

            persistent_separator = [m for m in separator_init if
                      m in separators]

            store_persistent_sep[-1].append(persistent_separator)

            persistent_clique = [m for m in clique_init if
                      m in clique]

            store_persistent_tetrahedron[-1].append(persistent_clique)

            motif_decay[g - u, u] += len(persistent_tri)
            separator_decay[g - u, u] += len(persistent_separator)
            clique_decay[g - u, u] += len(persistent_clique)
    plt.scatter(np.arange(1, time_length + 1), np.mean(motif_decay, axis=1), label = 'triangle') #/np.max(np.mean(motif_decay, axis=1))
    plt.scatter(np.arange(1, time_length + 1), np.mean(clique_decay, axis=1), label = 'tetrahedron') #/np.max(np.mean(clique_decay, axis=1))
    plt.scatter(np.arange(1, time_length + 1), np.mean(separator_decay, axis=1), label = 'separator') #/np.max(np.mean(separator_decay, axis=1))

    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.xlim(1, time_length + 50)
    plt.show()

    #from analysis import opt_R2, opt_pl

    break_points = opt_R2([np.mean(clique_decay, axis=1), np.mean(motif_decay, axis=1), np.mean(separator_decay, axis=1)], ['tetrahedron', 'triangle', 'separator'], 'NYSE')





    '''import pandas as pd
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt import risk_models
    from pypfopt import expected_returns

    # Read in price data
    df = pd.read_csv("tests/stock_prices.csv", parse_dates=True, index_col="date")

    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    # Optimise for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    ef.portfolio_performance(verbose=True)'''



    '''motif_decay = np.zeros((time_length, iterations))

    store_tri = []

    for u in range(iterations):

        print(u)

        W_init = np.loadtxt(open(file_list[u], "rb"), delimiter=",")

        print(W_init.shape)

        A_init, tri_init, separators_init, cliques_init, cliqueTree_init = TMFG(W_init, method='sq')

        tri_init = [set((tri_init[index, 0], tri_init[index, 1], tri_init[index, 2])) for index in range(tri_init.shape[0])]
        tri_temp = []
        for g in range(u, u + time_length):

            W = np.loadtxt(open(file_list[g], "rb"), delimiter=",")

            A, tri, separators, cliques, cliqueTree = TMFG(W, method='sq')

            tri = [set((tri[index, 0], tri[index, 1], tri[index, 2])) for index in
                        range(tri.shape[0])]

            tri_temp.append(tri)


            persistent = [m for m in tri_init if
                          m in tri]

            motif_decay[g - u, u] += len(persistent)
    plt.scatter(np.arange(time_length), np.mean(motif_decay, axis = 1))

    plt.show()'''

