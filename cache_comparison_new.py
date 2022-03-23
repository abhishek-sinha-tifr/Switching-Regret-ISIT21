# cache_comparison.py
""" Comparing the performances of different caching algorithms: FTPL, LRU, LFU and Marker 
Author: Samrat Mukhopadhyay"""
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from memoization import cached, CachingAlgorithmFlag as algo
import pickle

# Extract data from MovieLens dataset, N = 3952 
ratings = pd.read_csv("ratings.dat", sep = '::',names = ['col1','file','col3','time'], engine = 'python', nrows = 1000000)
new_ratings = ratings.sort_values(by = 'time')
new_ratings = new_ratings.reset_index(drop = True)
# # # print(ratings.tail(50))
files = new_ratings.T.iloc[1] #File indices in chronological order
T = 5 * 10 ** 4
N = 3952

# C = alpha * N, alpha typically 1%
alpha = 0.1
C = int(np.floor(N * alpha))

# Write the decorator functions for LRU and LFU:
@cached(max_size = C, algorithm = algo.LRU)
def f_lru(x):
    return x
@cached(max_size = C, algorithm = algo.LFU)
def f_lfu(x):
    return x
#@cached(max_size = C, algorithm = algo.FIFO)
#def f_fifo(x):
#    return x


# Run FTPL with static learning rate  
D = 1 # Switching cost D
eta_fixed = np.sqrt(2 * T / C) * (4 * np.pi * np.log(N / C)) ** (-1/4) # eta = sqrt(T(D+1)/C)(4 * pi * ln(N/C))^(-1/4)
gamma =  np.random.randn(N,)   # Sample the perturbation from standard Gaussian distribution
### Constant eta
X = np.zeros((N,T + 1)) # Initialize count vector
R_fixed, R_var, R_lru, R_lfu = [np.zeros((T + 1,1)) for _ in range(4)]# Total reward sequence
S_fixed, S_var, S_lru, S_lfu = [np.zeros((T + 1,1)) for _ in range(4)] # Total switching cost sequence
Q_fixed, Q_var, Q_lru, Q_lfu = [np.zeros((T,1)) for _ in range(4)]# Regret vector
lasty_fixed = lasty_var = np.zeros((N,))

for t in range(T):
    eta_t = np.sqrt(2 * (t + 1) / C) * (4 * np.pi * np.log(N / C)) ** (-1/4)
    print('time=', t)
    # X_per is the perturbed count vector
    X_per_fixed = X[:,t] + eta_fixed * gamma
    X_per_var = X[:,t] + eta_t * gamma
    
    # The FTPL predicted cache configuration
    y_fixed, y_var = np.zeros((N,)), np.zeros((N,))
    y_fixed[np.argsort(-X_per_fixed)[:C]] = 1 
    y_var[np.argsort(-X_per_var)[:C]] = 1
    
    # Request vector revealed
    x = np.zeros((N,)) 
    ft = int(files[t]) - 1
#    print(ft)
    x[ft] = 1
    X[:,t + 1] = X[:,t] + x
    
    # Update cache using LFU, LRU and FIFO 
    f_lru(ft)
#    print(f_lru.cache_info())
    f_lfu(ft)
#    print(f_lfu.cache_info())
#    f_fifo(ft)
    
    # Calculate instantaneous reward and switching cost for FTPL
    r_fixed = y_fixed[ft] 
    r_var = y_var[ft]
    if t == 0:
        s_fixed, s_var = 0, 0
    else:
        s_fixed = 0.5 * D * np.sum(np.abs(y_fixed - lasty_fixed)) 
        s_var = 0.5 * D * np.sum(np.abs(y_var - lasty_var))
    
    # Update total reward, switching cost and regret for FTPL
    lasty_fixed = y_fixed
    lasty_var = y_var
    cache = np.argsort(-X[:,t+1])
    cache = cache[:C]
#    cache_var = np.argsort(-X_var[:,t+1])
#    cache_var = cache_var[:C]
    opt = np.sum(X[cache,t+1])
    # Reward, Switching Cost and Regret update for FTPL
    R_fixed[t + 1], R_var[t + 1] = R_fixed[t] + r_fixed, R_var[t] + r_var
    S_fixed[t + 1], S_var[t + 1] = S_fixed[t] + s_fixed, S_var[t] + s_var
    Q_fixed[t], Q_var[t] = opt - R_fixed[t + 1] + S_fixed[t + 1], opt - R_var[t + 1] + S_var[t + 1]
    
    # Total Regret and Switching Cost update for LRU, LFU and FIFO
    R_lru[t + 1], R_lfu[t + 1] = f_lru.cache_info().hits, f_lfu.cache_info().hits
    S_lru[t + 1], S_lfu[t + 1] = f_lru.cache_info().misses, f_lfu.cache_info().misses
    Q_lru[t], Q_lfu[t] = opt - R_lru[t + 1] + S_lru[t + 1], opt - R_lfu[t + 1] + S_lfu[t + 1]

alphastr = str(alpha)    
f2 = open('caching_vars_alpha=' + alphastr + '.pickle', 'wb')
variables = dict(Hit = [R_fixed, R_var, R_lru, R_lfu], Switch = [S_fixed, S_var, S_lru, S_lfu], Regret = [Q_fixed, Q_var, Q_lru, Q_lfu], C = C, N = N, T = T)
pickle.dump(variables, f2)
#plotrange = np.arange(1,T+1).reshape(T,1)
#plotrange = plotrange[1 : T + 1 : 1000]
#l = len(plotrange)
#plt.figure(0)
#fig0 = plt.semilogy(plotrange, Q_fixed[plotrange].reshape(l,1) / plotrange, '-b', plotrange, Q_var[plotrange].reshape(l,1) / plotrange, '--b', plotrange, Q_lru[plotrange].reshape(l,1) / plotrange, '-.r', plotrange, Q_lfu[plotrange].reshape(l,1) / plotrange, '.g', linewidth = 2)
#plt.ylabel(r'$\frac{{R}_t}{t}$',fontsize = 20, rotation = 0)
#plt.xlabel(r'$t$', fontsize = 20)
#plt.grid()
##plt.ylim([0,0.3])
#plt.legend([r'FTPL, fixed $\eta_t$', r'FTPL, $\eta_t=\mathcal{O}(\sqrt{t/C})$', r'LRU', r'LFU'], fontsize = 20, loc = 'best')
#plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0), useMathText=True)
#plt.savefig('Regret_alpha=' + alphastr + '.pdf')
#
#plotrange = np.arange(2,T + 2).reshape(T,1)
#plotrange = plotrange[1 : T + 2: 1000]
#l = len(plotrange)
#plt.figure(1)
#fig1 = plt.semilogy(plotrange, S_fixed[plotrange].reshape(l,1) / plotrange, '-b', plotrange, S_var[plotrange].reshape(l,1) / plotrange, '--b', plotrange, S_lru[plotrange].reshape(l,1) / plotrange, '-.r', plotrange, S_lfu[plotrange].reshape(l,1) / plotrange, '.g', linewidth = 2)
#plt.ylabel(r'${FR}_t$',fontsize = 20, rotation = 0)
#plt.xlabel(r'$t$', fontsize = 20)
##plt.ylim([0,1])
#plt.legend([r'FTPL, fixed $\eta_t$', r'FTPL, $\eta_t=\mathcal{O}(\sqrt{t/C})$', r'LRU', r'LFU'], fontsize = 20, loc = 'best')
#plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0), useMathText=True)
#plt.grid()
#plt.savefig('Switching_cost_alpha=' + alphastr  + '.pdf')
#
#plt.figure(2)
#fig2 = plt.semilogy(plotrange, R_fixed[plotrange].reshape(l,1) / plotrange, '-b', plotrange, R_var[plotrange].reshape(l,1) / plotrange, '--b', plotrange, R_lru[plotrange].reshape(l,1) / plotrange, '-.r', plotrange, R_lfu[plotrange].reshape(l,1) / plotrange, '.g', linewidth = 2)
#plt.ylabel(r'Hit Rate',fontsize = 20, rotation = 0)
#plt.xlabel(r'$t$', fontsize = 20)
##plt.ylim([0,1])
#plt.legend([r'FTPL, fixed $\eta_t$', r'FTPL, $\eta_t=\mathcal{O}(\sqrt{t/C})$', r'LRU', r'LFU'], fontsize = 20, loc = 'best')
#plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0), useMathText=True)
#plt.grid()
#plt.savefig('Hit_rate_alpha=' + alphastr  + '.pdf')
#
#plt.show()
#f1 = open('caching_plots_alpha= ' + alphastr + '.pickle', 'wb')
#figs = [fig0, fig1, fig2]
#pickle.dump(figs, f1)
     