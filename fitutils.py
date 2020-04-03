import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.special as spe


def __fun(p, x, y, sx, sy):
        return (y - p[0]*x - p[1]) / (np.sqrt((p[0]*sx)**2 + sy**2+1e-14))

def __funzero(p, x, y, sx, sy):
        return (y - p[0]*x) / (np.sqrt((p[0]*sx)**2 + sy**2)+1e-14)

def __val2str(nsig, val):
    if nsig < 1:
        pre = -int(nsig - 2)
        return "{:.{}f}".format(val, pre)
    else:
        pre = 10**(int(nsig))
        return "{:.0f}".format(np.round(val / pre) * pre)

def __make_plot(x, y, dx, dy, p, s, re, l_ind, h_ind, cost, marker, markercolor,
                linecolor):
    nhull = 100
    tmp = np.array(np.unique(x,return_index=True))
    ind=tmp[1,:].astype(int)
    xplot = np.linspace(x[ind][0]-dx[ind][0], x[ind][-1]+dx[ind][-1], nhull)
    yplot = xplot * p[0] + p[1]
    plt.figure(figsize=(7,7))
    plt.errorbar(x, y, xerr=dx, yerr=dy, fmt='+', c=markercolor,
                 ecolor=markercolor)
    plt.plot(xplot, yplot,  color=linecolor)
    
    nd0 = np.log10(s[0]+1e-14)
    nd1 = np.log10(s[1]+1e-14)
    fmt_p0 = __val2str(nd0, p[0])
    fmt_s0 = __val2str(nd0, s[0])
    fmt_p1 = __val2str(nd1, p[1])
    fmt_s1 = __val2str(nd1, s[1])
    plt.title(r'Fit: y = (' + fmt_p0 + r' $\pm$ ' + fmt_s0
              + r') x + (' + fmt_p1 + r' $\pm$ ' + fmt_s1 + r')' + r'$\quad\chi^2=$'+"{:.{}f}".format(cost, 2))
    axipbi = xplot * np.ones((re.shape[0], xplot.shape[0])) * \
        np.meshgrid(np.ones(xplot.shape[0]), re[:, 0])[1] + \
        np.meshgrid(np.ones(xplot.shape[0]), re[:, 1])[1]
    saxipbi = np.sort(axipbi, axis=0)
    plt.plot(xplot, saxipbi[l_ind], '--', color=linecolor)
    plt.plot(xplot, saxipbi[h_ind], '--', color=linecolor)

def __make_plot_ord0(x, y, dx, dy, p, s, re, l_ind, h_ind, cost, marker, markercolor,
                linecolor):
    nhull = 100
    tmp = np.array(np.unique(x,return_index=True))
    ind=tmp[1,:].astype(int)
    xplot = np.linspace(x[ind][0]-dx[ind][0], x[ind][-1]+dx[ind][-1], nhull)
    yplot = xplot * p
    plt.figure(figsize=(7,7))
    plt.errorbar(x, y, xerr=dx, yerr=dy, fmt='+', c=markercolor,
                 ecolor=markercolor)
    plt.plot(xplot, yplot,  color=linecolor)
    
    nd0 = np.log10(s)
    fmt_p0 = __val2str(nd0, p)
    fmt_s0 = __val2str(nd0, s)
    plt.title(r'Fit: y = (' + fmt_p0 + r' $\pm$ ' + fmt_s0
              + r') x' + r'$\quad\chi^2=$'+"{:.{}f}".format(cost, 2))
    axipbi = xplot * np.ones((re.shape[0], xplot.shape[0])) * \
        np.meshgrid(np.ones(xplot.shape[0]), re)[1]
    saxipbi = np.sort(axipbi, axis=0)
    plt.plot(xplot, saxipbi[l_ind], '--', color=linecolor)
    plt.plot(xplot, saxipbi[h_ind], '--', color=linecolor)


def linfitxy(x, y, dx, dy, nbloop=500, nsigma=1, plot=True, marker='o',
             markercolor='tab:blue', linecolor='tab:orange',y0=False, **kwargs):
    if y0==True:
        return linfitxy_ord0(x, y, dx, dy, nbloop=500, nsigma=1, plot=True, marker='o',\
                 markercolor='tab:blue', linecolor='tab:orange', **kwargs)
    else:
        re = np.zeros((nbloop, 2))
        cost = np.zeros(nbloop)
        tmp1 = np.array(np.unique(x,return_index=True))
        ind=tmp1[1,:].astype(int)
        tmp2 = y[ind]
        tmp1 = tmp1[0,:]
        a0 = np.mean((tmp2[1:]-tmp2[:-1]) / (tmp1[1:]-tmp1[:-1]))
        indnozero = np.where(tmp1 != 0)[0]
        b0 = np.mean(tmp2[indnozero] / (a0 * tmp1[indnozero]))
        p0 = np.array([a0, b0])
        lst = opt.least_squares(__fun, p0, args=(x, y, dx, dy), **kwargs)
        popt = lst.x
        Npt=len(x)
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
            indsort = np.argsort(re[:, i])
            sorted_pt = re[indsort,i]
            p_low[i] = sorted_pt[l_ind]
            p_hig[i] = sorted_pt[h_ind]
        p = (p_low + p_hig) / 2
        s = p_hig - p
        cost = np.sqrt(np.sum(__fun(p, x, y, dx, dy)**2)/len(x))
        if plot:
            __make_plot(x, y, dx, dy, p, s, re, l_ind, h_ind, cost, marker, markercolor,
                        linecolor)
        return np.array([p[0], p[1], s[0], s[1]])

def linfitxy_ord0(x, y, dx, dy, nbloop=500, nsigma=1, plot=True, marker='o',
             markercolor='tab:blue', linecolor='tab:orange', **kwargs):
    re = np.zeros((nbloop))
    tmp1 = np.array(np.unique(x,return_index=True))
    ind=tmp1[1,:].astype(int)
    tmp2 = y[ind]
    tmp1 = tmp1[0,:]
    a0 = np.mean((tmp2[1:]-tmp2[:-1]) / (tmp1[1:]-tmp1[:-1]))
    p0 = a0
    lst = opt.least_squares(__funzero, p0, args=(x, y, dx, dy), **kwargs)
    popt = lst.x
    Npt=len(x)
    for i in range(nbloop):
        xi = rd.normal(x, dx)
        yi = rd.normal(y, dy)
        lst = opt.least_squares(__funzero, popt, args=(xi, yi, dx, dy))
        popt = lst.x
        re[i] = popt
    thres = 0.5 * spe.erfc(nsigma / np.sqrt(2))
    l_ind = int(1 + np.floor(nbloop * thres))
    h_ind = int(np.ceil(nbloop * (1 - thres)))

    indsort = np.argsort(re)
    sorted_pt = re[indsort]
    p_low = sorted_pt[l_ind]
    p_hig = sorted_pt[h_ind]
    
    p = (p_low + p_hig) / 2
    s = p_hig - p
    cost = np.sqrt(np.sum(__funzero(np.array([p]), x, y, dx, dy)**2)/len(x))
    if plot:
        __make_plot_ord0(x, y, dx, dy, p, s, re, l_ind, h_ind, cost, marker, markercolor,
                    linecolor)
    return np.array([p, s])
