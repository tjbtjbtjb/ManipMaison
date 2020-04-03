import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.special as spe


def __fun(p, x, y, sx, sy):
        return (y - p[0]*x - p[1]) / np.sqrt((p[0]*sx)**2 + sy**2)


def __val2str(nsig, val):
    if nsig < 1:
        pre = -int(nsig - 2)
        return "{:.{}f}".format(val, pre)
    else:
        pre = 10**(int(nsig))
        return "{:.0f}".format(np.round(val / pre) * pre)


def __make_plot(x, y, dx, dy, p, s, re, l_ind, h_ind, marker, markercolor,
                linecolor):
    nhull = 100
    xplot = np.linspace(x[0]-dx[0], x[-1]+dx[-1], nhull)
    yplot = xplot * p[0] + p[1]
    plt.figure()
    plt.errorbar(x, y, xerr=dx, yerr=dy, fmt=marker, c=markercolor,
                 ecolor=markercolor)
    plt.plot(xplot, yplot,  color=linecolor)
    nd0 = np.log10(s[0])
    nd1 = np.log10(s[1])
    fmt_p0 = __val2str(nd0, p[0])
    fmt_s0 = __val2str(nd0, s[0])
    fmt_p1 = __val2str(nd1, p[1])
    fmt_s1 = __val2str(nd1, s[1])
    plt.title(r'Fit: y = (' + fmt_p0 + r' $\pm$ ' + fmt_s0
              + r') x + (' + fmt_p1 + r' $\pm$ ' + fmt_s1 + r')')
    axipbi = xplot * np.ones((re.shape[0], xplot.shape[0])) * \
        np.meshgrid(np.ones(xplot.shape[0]), re[:, 0])[1] + \
        np.meshgrid(np.ones(xplot.shape[0]), re[:, 1])[1]
    saxipbi = np.sort(axipbi, axis=0)
    plt.plot(xplot, saxipbi[l_ind], '--', color=linecolor)
    plt.plot(xplot, saxipbi[h_ind], '--', color=linecolor)


def linfitxy(x, y, dx, dy, nbloop=500, nsigma=1, plot=True, marker='o',
             markercolor='tab:blue', linecolor='tab:orange', **kwargs):
    re = np.zeros((nbloop, 2))
    a0 = np.mean((y[1:]-y[:-1]) / (x[1:]-x[:-1]))
    b0 = np.mean(y - (a0 * x))
    p0 = np.array([a0, b0])
    lst = opt.least_squares(__fun, p0, args=(x, y, dx, dy), **kwargs)
    popt = lst.x
    for i in range(nbloop):
        xi = rd.normal(x, dx)
        yi = rd.normal(y, dy)
        lst = opt.least_squares(__fun, popt, args=(xi, yi, dx, dy))
        popt = lst.x
        re[i] = popt
    thres = 0.5 * spe.erfc(nsigma / np.sqrt(2))
    p_low = np.zeros(re.shape[1])
    p_hig = np.zeros(re.shape[1])
    l_ind = int(1 + np.floor(nbloop * thres))
    h_ind = int(np.ceil(nbloop * (1 - thres)))
    for i in range(re.shape[1]):
        sorted_pt = np.sort(re[:, i])
        p_low[i] = sorted_pt[l_ind]
        p_hig[i] = sorted_pt[h_ind]
    p = (p_low + p_hig) / 2
    s = p_hig - p
    if plot:
        __make_plot(x, y, dx, dy, p, s, re, l_ind, h_ind, marker, markercolor,
                    linecolor)
    return np.array([p[0], p[1], s[0], s[1]])
