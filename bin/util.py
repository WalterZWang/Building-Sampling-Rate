import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def generate_fig_name(fig_name):
    '''
    Generate the figure path and name
    To be used by plt.savefig function
    '''
    return '../docs/fig/{}'.format(fig_name)

def plot_FS(f, save_fig=False):

    dx = 1
    n = len(f)
    L = n
    x = np.arange(0, L, dx)

    fig, axs = plt.subplots(3, 2, figsize=(16,12))
    plt.subplots_adjust(wspace=0.9)

    # Compute Fourier series
    w_max = 96
    A = np.zeros(w_max)
    B = np.zeros(w_max)
    error = np.zeros(w_max)

    A[0] = np.sum(f * np.cos(2*np.pi*(0)*x/L)) *dx * 2/L
    fFS = A[0]/2
    error[0] = (sum((f - fFS)**2)/len(f))**0.5

    # plot the time domain
    axs[0, 0].sharex(axs[2, 0])
    axs[1, 0].sharex(axs[2, 0])

    axs[0, 0].plot(x,f,color='k',linewidth=1.5,label='Raw data')

    axs[2, 0].plot(x,f,color='k',linewidth=1.5,label='Raw data')
    for k in range(1, w_max):
        A[k] = np.sum(f * np.cos(2*np.pi*(k)*x/L)) *dx * 2/L # Inner product
        B[k] = np.sum(f * np.sin(2*np.pi*(k)*x/L)) *dx * 2/L
        fFS_k = A[k]*np.cos(2*np.pi*(k)*x/L) + B[k]*np.sin(2*np.pi*(k)*x/L)
        fFS = fFS + fFS_k
        error[k] = (sum((f - fFS)**2)/len(f))**0.5
        if k <= 5:
            axs[1, 0].plot(x,fFS_k,'--',label=f'FS: order={k}')
            axs[2, 0].plot(x,fFS,'--',label=f'FS: order={k}')    
        elif k%15 == 1:
            axs[1, 0].plot(x,fFS_k,'--',label=f'FS: order={k}')
            axs[2, 0].plot(x,fFS,'--',label=f'FS: order={k}')

    axs[2, 0].legend(loc='center left', bbox_to_anchor=(1, 1))

    xtick_frequency = 6 # unit: h 
    xticks = np.arange(0, len(f)+1, 60*xtick_frequency)
    axs[2, 0].set_xticks(xticks)
    axs[2, 0].set_xticklabels(xticks//60)
    axs[0, 0].set_ylabel('Measured Temperature\n[$^{o}$C]')
    axs[1, 0].set_ylabel('Fourier Components\n [$^{o}$C]')
    axs[2, 0].set_ylabel('Fourier Approximation\n[$^{o}$C]')
    axs[2, 0].set_xlabel('Time [h]')

    # plot the frequency domain
    axs[0, 1].sharex(axs[2, 1])
    axs[1, 1].sharex(axs[2, 1])

    # plot amplitude
    PSD = (A[1:]**2 + B[1:]**2)
    amplitude = PSD**0.5
    axs[0, 1].bar(np.arange(1,w_max), amplitude)
    axs[0, 1].set_yscale('log')

    # plot Accumulated percentage PSD
    PSD_cumsum = np.cumsum(PSD)
    PSD_cumsum_percentage = PSD_cumsum/PSD_cumsum[-1]*100
    axs[1, 1].plot(np.arange(1,w_max), PSD_cumsum_percentage)
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    axs[1, 1].yaxis.set_major_formatter(xticks)


    # plor error
    axs[2, 1].plot(np.arange(0,w_max), error)
    axs[2, 1].set_yscale('log')

    axs[2, 1].set_xticks([1, 12, 24, 48, 96])
    axs[2, 1].set_xticklabels([24, 2, 1, 0.5, 0.25])
    axs[0, 1].set_ylabel('Fourier Component \nAmplitude [$^{o}$C]')
    axs[1, 1].set_ylabel('Cumulative PSD')
    axs[2, 1].set_ylabel('Approximation error [$^{o}$C]')
    axs[2, 1].set_xlabel('Period [h]')
    
    if save_fig:
        plt.savefig(generate_fig_name(save_fig))
