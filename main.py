import statistics as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy as sp
from scipy.stats import poisson
from statsmodels.distributions.empirical_distribution import ECDF
from math import gamma
# 1st part


def first():
    sample_10_n = np.random.normal(loc=0, scale=1, size=10)
    sample_50_n = np.random.normal(loc=0, scale=1, size=50)
    sample_1000_n = np.random.normal(loc=0, scale=1, size=1000)

    # Normal distribution plots
    fig_n, axs_n = plt.subplots(1, 3, figsize=(12, 4))

    axs_n[0].hist(sample_10_n, density=True, alpha=0.5, bins=5)
    axs_n[0].plot(np.linspace(-10, 10, 1000),
                1/(np.sqrt(2*np.pi)*1)*np.exp(-(np.linspace(-10, 10, 1000)-0)**2/(2*1**2)))
    axs_n[0].set_title('Normal 10: mean=0, std=1, size=10')

    axs_n[1].hist(sample_50_n, density=True, alpha=0.5, bins=10)
    axs_n[1].plot(np.linspace(-10, 10, 1000),
                1/(np.sqrt(2*np.pi)*1)*np.exp(-(np.linspace(-10, 10, 1000)-0)**2/(2*1**2)))
    axs_n[1].set_title('Normal 50: mean=0, std=1, size=50')

    axs_n[2].hist(sample_1000_n, density=True, alpha=0.5, bins=30)
    axs_n[2].plot(np.linspace(-10, 10, 1000),
                1/(np.sqrt(2*np.pi)*1)*np.exp(-(np.linspace(-10, 10, 1000)-0)**2/(2*1**2)))
    axs_n[2].set_title('Normal 1000: mean=0, std=1, size=1000')
    plt.show()

    # Cauchy distribution plots
    loc_c = 0.0
    scale_c = 1.0
    sample_10_c = np.random.standard_cauchy(size=10)*scale_c + loc_c
    sample_50_c = np.random.standard_cauchy(size=50)*scale_c + loc_c
    sample_50_trunc = sample_50_c[(sample_50_c > -10) & (sample_50_c < 10)]
    sample_1000_c = np.random.standard_cauchy(size=1000)*scale_c + loc_c
    sample_1000_trunc = sample_1000_c[(
        sample_1000_c > -10) & (sample_1000_c < 10)]

    fig_c, axs_c = plt.subplots(1, 3, figsize=(12, 4))

    axs_c[0].hist(sample_10_c, density=True, alpha=0.5, bins=10)
    axs_c[0].plot(np.linspace(-10, 10, 1000), (1/np.pi)
                  * (1/(np.linspace(-10, 10, 1000)**2)))
    axs_c[0].set_title('Cauchy 10: mean=0, std=1, size=10')
    axs_c[0].set_ylim([0, 1])

    axs_c[1].hist(sample_50_trunc, density=True, alpha=0.5, bins=20)
    axs_c[1].plot(np.linspace(-10, 10, 1000), (1/np.pi)
                  * (1/(np.linspace(-10, 10, 1000)**2)))
    axs_c[1].set_title('Cauchy 50: mean=0, std=1, size=50')
    axs_c[1].set_ylim([0, 1])

    axs_c[2].hist(sample_1000_trunc, density=True, alpha=0.5, bins=30)
    axs_c[2].plot(np.linspace(-10, 10, 1000), (1/np.pi)
                  * (1/(np.linspace(-10, 10, 1000)**2)))
    axs_c[2].set_title('Cauchy 1000: mean=0, std=1, size=1000')
    axs_c[2].set_ylim([0, 1])
    plt.show()

    # Laplace distribution plots
    sample_10_l = np.random.laplace(loc=0, scale=1/np.sqrt(2), size=10)
    sample_50_l = np.random.normal(loc=0, scale=1/np.sqrt(2), size=50)
    sample_1000_l = np.random.normal(loc=0, scale=1/np.sqrt(2), size=1000)

    fig_l, axs_l = plt.subplots(1, 3, figsize=(12, 4))

    axs_l[0].hist(sample_10_l, density=True, alpha=0.5, bins=5)
    axs_l[0].plot(np.linspace(-10, 10, 1000),
                1/(np.sqrt(2))*np.exp(-np.sqrt(2)*np.abs(np.linspace(-10, 10, 1000))))
    axs_l[0].set_title('Laplace 10: mean=0, std=1/sqrt(2), size=10')

    axs_l[1].hist(sample_50_l, density=True, alpha=0.5, bins=10)
    axs_l[1].plot(np.linspace(-10, 10, 1000),
                1/(np.sqrt(2))*np.exp(-np.sqrt(2)*np.abs(np.linspace(-10, 10, 1000))))
    axs_l[1].set_title('Laplace 50: mean=0, std=1/sqrt(2), size=10')

    axs_l[2].hist(sample_1000_l, density=True, alpha=0.5, bins=20)
    axs_l[2].plot(np.linspace(-10, 10, 1000),
                1/(np.sqrt(2))*np.exp(-np.sqrt(2)*np.abs(np.linspace(-10, 10, 1000))))
    axs_l[2].set_title('Laplace 1000: mean=0, std=1/sqrt(2), size=10')

    plt.show()

    # Poisson distribution plots

    sample_10_p = np.random.poisson(lam=10, size=10)
    sample_50_p = np.random.poisson(lam=10, size=50)
    sample_1000_p = np.random.poisson(lam=10, size=1000)

    fig_p, axs_p = plt.subplots(1, 3, figsize=(12, 4))

    axs_p[0].hist(sample_10_p, density=True, alpha=0.5, bins=5)
    axs_p[0].plot(np.arange(0, 10), poisson.pmf(np.arange(0, 10), 10))
    axs_p[0].set_title('Poisson 10: lambda = 10, size=10')
    axs_p[0].set_ylim([0, 1])

    axs_p[1].hist(sample_50_p, density=True, alpha=0.5, bins=10)
    axs_p[1].plot(np.arange(0, 50), poisson.pmf(np.arange(0, 50), 10))
    axs_p[1].set_title('Poisson 50: lambda = 10, size=50')
    axs_p[1].set_ylim([0, 1])

    axs_p[2].hist(sample_1000_p, density=True, alpha=0.5, bins=30)
    axs_p[2].plot(np.arange(0, 1000), poisson.pmf(np.arange(0, 1000), 10))
    axs_p[2].set_title('Poisson 1000: lambda = 10, size=1000')
    axs_p[2].set_ylim([0, 1])

    plt.show()
    # uniform distribution plots

    sample_10_u = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=10)
    sample_50_u = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=50)
    sample_1000_u = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=1000)

    fig_u, axs_u = plt.subplots(1, 3, figsize=(12, 4))
    linspace = [1/(2*np.sqrt(3))]*1000

    axs_u[0].hist(sample_10_u, density=True, alpha=0.5, bins=5)
    axs_u[0].plot(np.linspace(-np.sqrt(3), np.sqrt(3), 1000), linspace)
    axs_u[0].set_title('Uniform 10: lambda = 10, size=10')
    axs_u[0].set_ylim([0, 1])

    axs_u[1].hist(sample_50_u, density=True, alpha=0.5, bins=20)
    axs_u[1].plot(np.linspace(-np.sqrt(3), np.sqrt(3), 1000), linspace)
    axs_u[1].set_title('Uniform 10: lambda = 10, size=50')
    axs_u[1].set_ylim([0, 1])

    axs_u[2].hist(sample_1000_u, density=True, alpha=0.5, bins=30)
    axs_u[2].plot(np.linspace(-np.sqrt(3), np.sqrt(3), 1000), linspace)
    axs_u[2].set_title('Uniform 10: lambda = 10, size=1000')
    axs_u[2].set_ylim([0, 1])

    plt.show()

# first()


def generate_sample(name, n):
    if (name == "Normal"):
        sample = np.random.normal(loc=0, scale=1, size=n)
    elif (name == "Cauchy"):
        loc_c = 0.0
        scale_c = 1.0
        sample = np.random.standard_cauchy(size=n)*scale_c + loc_c
    elif (name == "Laplace"):
        sample = np.random.laplace(loc=0, scale=1/np.sqrt(2), size=n)
    elif (name == "Poisson"):
        sample = np.random.poisson(lam=10, size=n)
    elif (name == "Uniform"):
        sample = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=n)
    return sample


def second(name, n):
    iters = 1000
    mean = 0
    med = 0
    z_r = 0
    z_q = 0
    z_tr = 0
    mean_2 = 0
    med_2 = 0
    z_r_2 = 0
    z_q_2 = 0
    z_tr_2 = 0
    for i in range(iters):
        # sorting
        sample = generate_sample(name=name, n=n)
        sample.sort()

        # average mean
        tmp = sample.mean()
        mean += tmp
        mean_2 += tmp ** 2

        # median
        tmp = np.median(sample)
        med += tmp
        med_2 += tmp ** 2

        # z_r
        tmp = (sample[0] + sample[-1])/2
        z_r = tmp
        z_r_2 = tmp ** 2

        # quantiles
        tmp = (np.quantile(sample, 0.25) + np.quantile(sample, 0.75)) / 2
        z_q += tmp
        z_q_2 += tmp ** 2

        # trimmed mean
        r = n // 4
        tmp = sum(sample[r:-r]) / (n - 2 * r)
        z_tr += tmp
        z_tr_2 += tmp ** 2

    mean /= iters
    med /= iters
    z_r /= iters
    z_q /= iters
    z_tr /= iters
    mean_2 /= iters
    med_2 /= iters
    z_r_2 /= iters
    z_q_2 /= iters
    z_tr_2 /= iters

    d_mean = mean_2 - mean ** 2
    d_med = med_2 - med ** 2
    d_z_r = z_r_2 - z_r ** 2
    d_z_q = z_q_2 - z_q ** 2
    d_z_tr = z_tr_2 - z_tr ** 2

    return tuple(map(lambda x: round(x, 4), (mean, med, z_r, z_q, z_tr,
           d_mean, d_med, d_z_r, d_z_q, d_z_tr,
           mean - np.sqrt(d_mean),
           med - np.sqrt(d_med),
           z_r - np.sqrt(d_z_r),
           z_q - np.sqrt(d_z_q),
           z_tr - np.sqrt(d_z_tr),
           mean + np.sqrt(d_mean),
           med + np.sqrt(d_med),
           z_r + np.sqrt(d_z_r),
           z_q + np.sqrt(d_z_q),
           z_tr + np.sqrt(d_z_tr))))



# for name in ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']:
#     for n in [10, 100, 1000]:
#         res = second(name, n)
#         print(f'{name}, n = {n}', ' & '.join(map(str, res)))
#     print()

#------------------

# Tukey boxplot
def third_boxplot(name, n):
    
    sample = generate_sample(name = name, n = n)
    fig, ax = plt.subplots(1, 1)
    sb.boxplot(data = sample, orient='h', ax = ax)
    ax.set(xlabel='x', ylabel='n')
    ax.set(yticklabels=[len(sample)])
    ax.set_title(name)
    plt.show()

def calc_outlier(name, n, iters=1000):
    sample = generate_sample(name = name, n = n)
    num = 0
    for i in range(iters):
        q1 = np.quantile(sample, 0.25)
        q3 = np.quantile(sample, 0.75)
        iqr = q3 - q1
        x1 = q1 - 1.5 * iqr
        x2 = q3 + 1.5 * iqr
        num += np.count_nonzero((sample < x1) | (x2 < sample)) / n
    return round(num / iters, 2)



# for name in ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']:
#     for n in [20, 100]:
#         third_boxplot(name, n)

# for name in ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']:
#     for n in [20, 100]:
#         print(f'{name} {n}:', calc_outlier(name, n))




#Empirical distribution function

def make_edf(name, datasets, cdf, x): 
    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 5))
    fig.suptitle(name)
    for i, data in enumerate(datasets):
        y1 = ECDF(data)(x)
        y2 = cdf(x)
        axes[i].plot(x, y1)
        axes[i].plot(x, y2)
        axes[i].set_title(f'n = {len(data)}')
    plt.show()

def select_cdf(name):
    if (name == "Normal"):
        cdf = sp.stats.norm.cdf
    elif (name == "Cauchy"):
        cdf = sp.stats.cauchy.cdf
    elif (name == "Laplace"):
        cdf = lambda x: sp.stats.laplace.cdf(x, 0, 1 / np.sqrt(2))
    elif (name == "Poisson"):
        cdf = lambda x: sp.stats.poisson.cdf(x, 10)
    elif (name == "Uniform"):
        cdf = lambda x: sp.stats.uniform.cdf(x, -np.sqrt(3), 2 * np.sqrt(3))
    return cdf

def select_pdf(name):
    if (name == "Normal"):
        pdf = sp.stats.norm.pdf
    elif (name == "Cauchy"):
        pdf = sp.stats.cauchy.pdf
    elif (name == "Laplace"):
        pdf = lambda x: sp.stats.laplace.pdf(x, 0, 1 / np.sqrt(2))
    elif (name == "Poisson"):
        pdf = lambda x: 10 ** x * np.exp(-10) / gamma(x + 1)
    elif (name == "Uniform"):
        pdf = lambda x: sp.stats.uniform.pdf(x, -np.sqrt(3), 2 * np.sqrt(3))
    return pdf


# for name in ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']:
#     if name != 'Poisson':
#         make_edf(name, [generate_sample(name, n) for n in[20, 60, 100]], select_cdf(name), np.linspace(-4, 4, 100))
#     else:
#         make_edf(name, [generate_sample(name, n) for n in[20, 60, 100]], select_cdf(name), np.linspace(6, 14, 100))                    
                     

#kernel density estimation

def make_kde(name, data, pdf, x):
    scales = [0.5, 1.0, 2.0]
    fig, ax = plt.subplots(1, len(scales), figsize=(12, 4))
    fig.suptitle(f'{name}, n = {len(data)}')
    for i, scale in enumerate(scales):
        sb.kdeplot(data, ax=ax[i], bw_method='silverman', bw_adjust=scale, label='kde')
        ax[i].set_xlim([x[0], x[-1]])
        ax[i].set_ylim([0, 1])
        ax[i].plot(x, [pdf(xk) for xk in x], label='pdf')
        ax[i].legend()
        ax[i].set_title(f'h={str(scale)}*$h_n$')
    plt.show()
for name in ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']:
    for n in [20, 60, 100]:
        if name != 'Poisson':
            make_kde(name, generate_sample(name, n), select_pdf(name), np.linspace(-4, 4, 100))
        else:
            make_kde(name, generate_sample(name, n), select_pdf(name), np.linspace(6, 14, 100))                    
          