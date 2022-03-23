# Cache comparison data plot
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure as fig
alphavec = [0.01, 0.1]

R_fixed, R_var, R_lru, R_lfu, S_fixed, S_var, S_lru, S_lfu, Q_fixed, Q_var, Q_lru, Q_lfu = [{alpha:[] for alpha in alphavec} for _ in range(12)]
for alpha in [0.01, 0.1]:
#    alpha = 0.1
    alphastr = str(alpha)
    
    #with open('caching_plots_alpha= ' + alphastr + '.pickle', 'rb') as fid:
    #    figs = pkl.load(fid)
        
    with open('caching_vars_alpha=' + alphastr + '.pickle', 'rb') as fid:
        variables = pkl.load(fid)
    
    R_fixed[alpha], R_var[alpha], R_lru[alpha], R_lfu[alpha] = variables['Hit']
    S_fixed[alpha], S_var[alpha], S_lru[alpha], S_lfu[alpha] = variables['Switch']
    Q_fixed[alpha], Q_var[alpha], Q_lru[alpha], Q_lfu[alpha] = variables['Regret']
    
C = variables['C']
N = variables['N']
T = variables['T']

alpha = 0.01
alphastr = str(alpha)
plotrange = np.arange(1,T+1).reshape(T,1)
plotrange = plotrange[1 : T + 1 : 1000]
l = len(plotrange)
plt.figure(0)
fig(figsize = (9,6))
plt.semilogy(plotrange, Q_fixed[alpha][plotrange].reshape(l,1) / plotrange, '-b', plotrange, Q_var[alpha][plotrange].reshape(l,1) / plotrange, '--g', plotrange, Q_lru[alpha][plotrange].reshape(l,1) / plotrange, '-.r', plotrange, Q_lfu[alpha][plotrange].reshape(l,1) / plotrange, '.k', linewidth = 4)
#plt.hold(True) 
#plt.ylabel(r'$\frac{{R}_t}{t}$',fontsize = 30, rotation = 0)
#plt.xlabel(r'$t$', fontsize = 30)
##plt.ylim([0,0.3])
#plt.rc('font', size = 30)
#plt.legend([r'FTPL, fixed $\eta_t$', r'FTPL, $\eta_t=\mathcal{O}(\sqrt{t/C})$', r'LRU', r'LFU'], fontsize = 20, loc = 'best')
#plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0), useMathText=True)
plt.tick_params(labelleft = False, labelbottom = False)
#plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
plt.grid()
plt.tight_layout()
plt.savefig('Regret_without_labels_alpha=' + alphastr + '.pdf')

plotrange = np.arange(2,T + 2).reshape(T,1)
plotrange = plotrange[1 : T + 2: 1000]
l = len(plotrange)
plt.figure(1)
fig(figsize = (9,6))
plt.semilogy(plotrange, S_fixed[alpha][plotrange].reshape(l,1) / plotrange, '-b', plotrange, S_var[alpha][plotrange].reshape(l,1) / plotrange, '--g', plotrange, S_lru[alpha][plotrange].reshape(l,1) / plotrange, '-.r', plotrange, S_lfu[alpha][plotrange].reshape(l,1) / plotrange, '.k', linewidth = 4)
#plt.ylabel(r'${FR}_t$',fontsize = 30, rotation = 0)
#plt.xlabel(r'$t$', fontsize = 30)
#plt.ylim([0,1])
#plt.rc('font', size = 30)
#plt.legend([r'FTPL, fixed $\eta_t$', r'FTPL, $\eta_t=\mathcal{O}(\sqrt{t/C})$', r'LRU', r'LFU'], fontsize = 20, loc = 'center right')
#plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0), useMathText=True)
plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
plt.grid()
plt.tight_layout()
plt.savefig('Switching_cost_without_labels_alpha=' + alphastr  + '.pdf')

plt.figure(2)
fig(figsize = (9,6))
plt.plot(plotrange, R_fixed[alpha][plotrange].reshape(l,1) / plotrange, '-b', plotrange, R_var[alpha][plotrange].reshape(l,1) / plotrange, '--g', plotrange, R_lru[alpha][plotrange].reshape(l,1) / plotrange, '-.r', plotrange, R_lfu[alpha][plotrange].reshape(l,1) / plotrange, '.k', linewidth = 4)
#plt.ylabel(r'Hit Rate',fontsize = 30, rotation = 0)
#plt.xlabel(r'$t$', fontsize = 30)
#plt.ylim([0,1])
#plt.legend([r'FTPL, fixed $\eta_t$', r'FTPL, $\eta_t=\mathcal{O}(\sqrt{t/C})$', r'LRU', r'LFU'], fontsize = 20, loc = 'best')
#plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0), useMathText=True)
plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
plt.grid()
#plt.rc('font', size = 30)
plt.tight_layout()
plt.savefig('Hit_rate_without_labels_alpha=' + alphastr  + '.pdf')

plt.show()

#f1 = open('caching_plots_without_legend_alpha= ' + alphastr + '.pickle', 'wb')
#figs = [fig0, fig1, fig2]
#pkl.dump(figs, f1)