import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import scipy
import scipy.stats as st
import matplotlib.dates as mdates

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                  AutoMinorLocator)
plt.rc('font', family='serif')
plt.rc('text', usetex=True)#set to true if available?
plt.ion()

dtFmt=mdates.DateFormatter('%b-%Y')

def makefig(shape=(1,1), projection=None, sharex=None, sharey=None, **kwargs):
    '''
    shape = tuple like (1,1) (2,1) (1,2) etc
    '''
    fig = plt.figure(**kwargs)
    if shape == (1,1):
        ax = fig.add_subplot(111, projection = projection,sharex=sharex, sharey=sharey )
    else:
        #shape[0] = # of rows
        #shape[1] = # of columns
        gs = fig.add_gridspec(shape[0], shape[1], hspace=0, wspace=0)
        ax = gs.subplots(sharex='col', sharey='row')
    return fig,ax


def figure_check(fig=None, ax= None,shape=(1,1), makefigax=True, projection=None, sharex=None, sharey=None, **kwargs):
    if makefigax:
        if fig is None:
            if ax is None :
                fig, ax = makefig(shape=shape, projection=projection, sharex=sharex, sharey=sharey)
            else:
                ax = plt.gca()
        else:
            ax = fig.add_subplot(111, projection = projection,sharex=sharex, sharey=sharey )        
    return fig, ax

def set_aesthetics(fig = None, ax=None,makefigax=False, **kwargs):
    fig, ax = figure_check(fig=fig, ax=ax, makefigax=makefigax)
    if kwargs.get('xlim'):
        ax.set_xlim(kwargs['xlim'])
    if kwargs.get('ylim'):
        ax.set_xlim(kwargs['xlim'])
    if kwargs.get('xlabel'):
        ax.set_xlabel(kwargs['xlabel'])        
    if kwargs.get('xlabel'):
        ax.set_ylabel(kwargs['ylabel'])
    if kwargs.get('aspect'):
        ax.set_aspect(kwargs['aspect'])
    if kwargs.get('title'):
        ax.set_title(kwargs['title'])
    if kwargs.get('datetime'):
        ax.xaxis.set_major_formatter(dtFmt)
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 20)

    plt.minorticks_on()

            
def scatter(x,y, plotdata=True,  fig=None, ax=None, makefigax=True,
            #for plt.scatter
            facecolor=None, cmap='plasma', edgecolor='k', 
            alpha=1, color='k', marker='o', 
            s=5,   vmin=None, vmax=None, label='',
            ccode=None,
            #for plot2dhist
            binlabel='',bin_y=False, 
            lim=True, setplotlims=True, bin_stat_y='mean',
            size_y_bin=0.25, counting_thresh=5, percentiles = False,            
            #for plt.plot
            linecolor='r', linewidth=3,
            #for set_aesthetics
            xlabel='', ylabel='', aspect='auto', 
            xlim = [], ylim = []):      
    fig, ax = figure_check(fig=fig, ax=ax, makefigax=makefigax)
    print(fig, ax)
    if plotdata:
        ax.scatter(x,y, facecolor=facecolor, 
            cmap=cmap,
            edgecolor=edgecolor, 
            alpha=alpha,
            color=color, 
            c = ccode,
            marker=marker, 
            s=s, 
            vmin=vmin, 
            vmax=vmax,
            label=label)
    set_aesthetics(xlabel=xlabel, 
            ylabel=ylabel,  
            aspect=aspect, 
            xlim = xlim, 
            ylim = ylim,
            fig=fig, ax=ax, makefigax=False)
    if bin_y:
        outs = plot2dhist(x,y,data=False, ax=ax, fig=fig, makefigax=False,
            label=binlabel,bin_y=bin_y, 
            lim=lim, setplotlims=setplotlims, bin_stat_y=bin_stat_y,
            size_y_bin=size_y_bin, counting_thresh=counting_thresh, percentiles = percentiles
            )
        if plotdata:
            if percentiles:
                ax.plot(outs['xmid'], outs['bins16'], linecolor+'-.', linewidth=linewidth )
                ax.plot(outs['xmid'], outs['bins84'], linecolor+'-.', linewidth=linewidth)
            ax.plot(outs['xmid'], outs['avg_y'], linewidth=linewidth, color=linecolor, label=binlabel)
            return outs
        else:
            if plotdata:
                ax.plot(outs['xmid'], outs['avg_y'], linewidth=linewidth, color=linecolor,label=binlabel)        
        return outs
def error(x,y, plotdata=True,  fig=None, ax=None, makefigax=True,
            #for plt.error
            xerr=[], yerr=[], elinewidth=0.1,
            capsize=3, capthick=0.1,ecolor='k', 
            fmt='none',  label='',
            #for plot2dhist
            binlabel='',bin_y=False,lim=True, 
            setplotlims=True, bin_stat_y='mean',
            size_y_bin=0.25, counting_thresh=5, 
            percentiles = False,
            #for plt.plot
            linecolor='r',linewidth=3,
            #for set_aesthetics
            xlabel='', ylabel='', aspect='auto',
            xlim = [], ylim = []):
    fig, ax = figure_check(fig=fig, ax=ax, makefigax=makefigax)  
    if plotdata:
        ax.errorbar(x,y, xerr=xerr, 
            yerr=yerr,
            elinewidth=elinewidth,
            capsize=capsize, 
            capthick=capthick, 
            ecolor=ecolor, 
            fmt=fmt, label=label
            )                
    set_aesthetics(xlabel=xlabel, 
            ylabel=ylabel,  
            aspect=aspect, 
            xlim = xlim, 
            ylim = ylim, fig = fig, ax=ax, makefigax=makefigax)
    if bin_y:
        outs = plot2dhist(x,y,data=False,  ax=ax, fig=fig, makefigax=False,
            label=binlabel,bin_y=bin_y, 
            lim=lim, setplotlims=setplotlims, bin_stat_y=bin_stat_y,
            size_y_bin=size_y_bin, counting_thresh=counting_thresh, percentiles = percentiles,)
        if plotdata:
            if percentiles:
                ax.plot(outs['xmid'], outs['bins16'], linecolor+'-.', linewidth=linewidth )
                ax.plot(outs['xmid'], outs['bins84'], linecolor+'-.', linewidth=linewidth)
            ax.plot(outs['xmid'], outs['avg_y'], linewidth=linewidth, color=linecolor, label=binlabel)
            return outs
        else:
            if plotdata:
                ax.plot(outs['xmid'], outs['avg_y'], linewidth=linewidth, color=linecolor,label=binlabel)        
        return outs    
def finite(arr):    
    fin = np.isfinite(arr) 
    return fin
def filter_copy(arr, filter):
    arr_copy = np.copy(arr[filter])
    return arr_copy
def get_plotting_bounds(quant):
    return [np.sort(quant)[int(0.01*len(quant))], np.sort(quant)[int(0.99*len(quant))]]
def get_within_bounds(quant, mn,mx):
    return (quant<mx) & (quant>mn)
def plot2dhist(x,y,
               nx=200,
               ny=200, 

               bin_y=False, 
               bin_stat_y = 'mean',  
               ybincolsty='r-',
               plotlines=True,
               ybincolsty_perc='r-.',
               nbins=25, 
               size_y_bin=0,
               bin_quantity=[],
               percentiles=False,
               counting_thresh=20, 
               linewid=2, 


               ccode= [], 
               ccode_stat=np.nanmedian,               
               ccodename = '', 
               ccodelim=[], 
               ccode_bin_min=20, 
               cmap='plasma', 
               show_cbar=True,
               
               
               dens_scale=0.3,
               label='', 
               zorder=10,
               nan=True, 
               data=True, 
               fig=None,
               makefigax=True, 
               ax=None, 
               xlim=[0,0], 
               ylim=[0,0], 
               xlabel='', 
               ylabel='', 
               lim=False, 
               setplotlims=False, 
               aspect='auto'):
    if type(x)!=np.array:
        x = np.copy(np.array(x))
    if type(y)!=np.array:
        y = np.copy(np.array(y))
    if len(ccode)!=0:
        if type(ccode)!=np.array:
            ccode=np.copy(np.array(ccode))
    if len(bin_quantity) != 0:
        if type(bin_quantity)!=np.array:
            bin_quantity = np.copy(np.array(bin_quantity))
    print(makefigax)
    fig,ax = figure_check(fig=fig, ax=ax, makefigax=makefigax)
    if nan:
        fin_  = finite(x) & finite(y)         
        if len(ccode)!=0:
            fin_ = finite(x) & finite(y) &  finite(ccode) 
            ccode= filter_copy(ccode, fin_)
        x = filter_copy(x,fin_)
        y =filter_copy(y,fin_)
        if len(bin_quantity) != 0:
            bin_quantity = filter_copy(bin_quantity,fin_)
    if xlim[0] == 0 and xlim[1]==0 and ylim[0]==0 and ylim[1]==0:
        xlim = get_plotting_bounds(x)
        ylim = get_plotting_bounds(y)        
    elif xlim[0] !=0  and xlim[1] !=0 and ylim[0]==0 and ylim[1]==0:
        ylim = get_plotting_bounds(y) 
    elif xlim[0] ==0 and xlim[1] == 0 and ylim[0]!=0 and ylim[1] !=0:
        xlim = get_plotting_bounds(x)
    if lim:
        limited = ( (get_within_bounds(x, xlim[0], xlim[1])) &
                            (get_within_bounds(y, ylim[0], ylim[1]))  )
        x = filter_copy(x, limited)
        y =filter_copy(y, limited)
        if len(bin_quantity) != 0:
            bin_quantity = filter_copy(bin_quantity, limited)
        if len(ccode) !=0:
            ccode=filter_copy(ccode, limited)
            
    hist, xedges, yedges = np.histogram2d(x,y,bins = (int(nx),int(ny)), range=[xlim, ylim ])
    extent= [np.min(xedges),np.max(xedges),np.min(yedges),np.max(yedges)]
    set_aesthetics(xlim=xlim, ylim=ylim, aspect=aspect, xlabel=xlabel, ylabel=ylabel, fig=fig, ax=ax, makefigax=False)

    if len(ccode)==0:        
        if data:

            ax.imshow((hist.transpose())**dens_scale, cmap='gray_r',extent=extent,origin='lower',
                        aspect=aspect,alpha=0.9)                
        if bin_y:
            if size_y_bin !=0:
                nbins = int( (xlim[1]-xlim[0])/size_y_bin)
            else:
                size_y_bin = round((xlim[1]-xlim[0])/(nbins), 2)
            avg_y, xedges, binnum = scipy.stats.binned_statistic(x,y, statistic=bin_stat_y, bins = nbins,range=xlim)
            count_y, xedges, binnum = scipy.stats.binned_statistic(x,y,statistic='count', bins = nbins,range=xlim )
            good_y = np.where(count_y>=counting_thresh)[0]
            xmid = (xedges[1:]+xedges[:-1])/2
            if percentiles:
                bins84 = []
                bins16 = []
                for i in range(len(xedges)-1):
                    binned_ = np.where((x>xedges[i])&(x<xedges[i+1]))[0]
                    if i in good_y:
                        bins16.append(np.percentile(y[binned_], 16))
                        bins84.append(np.percentile(y[binned_], 84))
                    else:
                        bins16.append(np.nan)
                        bins84.append(np.nan)                        
                bins84 = np.array(bins84)
                bins16 = np.array(bins16)
            if len(bin_quantity) !=0:
                avg_quant, _xedges, _binnum = scipy.stats.binned_statistic(x,bin_quantity, statistic=bin_stat_y, bins = nbins,range=xlim)
                avg_quant = avg_quant[good_y]
            else:
                avg_quant = bin_quantity
            if ax:
                if plotlines:
                    ax.plot(xmid[good_y], avg_y[good_y], ybincolsty, linewidth=linewid, label=label, zorder=zorder)
                    if percentiles:
                        ax.plot(xmid[good_y], bins84[good_y], ybincolsty_perc, linewidth=linewid, label=label, zorder=zorder)
                        ax.plot(xmid[good_y], bins16[good_y], ybincolsty_perc, linewidth=linewid, label=label, zorder=zorder)
                if percentiles:
                    return {'fig':fig, 'ax':ax, 'xmid':xmid[good_y], 'avg_y':avg_y[good_y], 'avg_quant':avg_quant, 'bins16':bins16[good_y], 'bins84':bins84[good_y]}
                return {'fig':fig,'ax': ax, 'xmid':xmid[good_y], 'avg_y':avg_y[good_y],'avg_quant': avg_quant}
            else:
                if plotlines:
                    plt.plot(xmid[good_y], avg_y[good_y],ybincolsty, linewidth=linewid, label=label, zorder=zorder)
                    if percentiles: 
                        plt.plot(xmid[good_y], bins84[good_y], ybincolsty_perc, linewidth=linewid, label=label, zorder=zorder)
                        plt.plot(xmid[good_y], bins16[good_y], ybincolsty_perc, linewidth=linewid, label=label, zorder=zorder)
                        
                if percentiles: 
                    return {'fig':fig,'ax':ax,'xmid': xmid[good_y], 'avg_y':avg_y[good_y], 'avg_quant':avg_quant, 'bins16':bins16[good_y], 'bins84':bins84[good_y]      }            
                return {'fig':fig, 'ax':ax, 'xmid':xmid[good_y], 'avg_y':avg_y[good_y], 'avg_quant':avg_quant}
    else:
        ccode_avgs = np.zeros_like(hist)
        for i in range(len(xedges)-1):
            for j in range(len(yedges)-1):
                val_rang = np.where( (x>=xedges[i]) &(x<xedges[i+1]) &
                                     (y>=yedges[j]) & (y<yedges[j+1]))[0]
                if val_rang.size >= ccode_bin_min:
                    ccode_avgs[i,j] = ccode_stat(ccode[val_rang])
                else:
                    ccode_avgs[i,j]= np.nan
        if len(ccodelim) ==2:
            mn, mx = ccodelim
        else:
            mn, mx = np.nanmin(ccode_avgs), np.nanmax(ccode_avgs)       
        im = ax.imshow((ccode_avgs.transpose()), cmap=cmap,extent=extent,origin='lower',
                aspect=aspect,alpha=0.9, vmin=mn, vmax=mx)#, norm=colors.PowerNorm(gamma=1/2)) 
        if show_cbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(ccodename, fontsize=20)
            cbar.ax.tick_params(labelsize=20)
        return {'im':im, 'fig':fig,'ax': ax, 'ccode_avgs':ccode_avgs}
def plothist(x, bins=10, range=(), linewidth=1, datetime=False,
             cumulative=False, reverse=False, ylim=False,
             label='',linestyle='--', c='k',
             density=False, xlabel='',ylabel='Counts', normed=False, norm0= False, normval=None,
             integrate = False,fig=None, ax=None, makefigax=True):
    '''
    Needs to be pared down/refactored still 9/14-CA
    '''
    fig, ax = figure_check(fig=fig, ax=ax, makefigax=makefigax)
    if range==():
        range = (np.min(x), np.max(x))
    if datetime:    
        cnts, bins = np.histogram(mdates.date2num(x), bins=bins, density=density)

    else:
        cnts, bins = np.histogram(x, bins=bins, range=range, density=density)
    bncenters = (bins[:-1]+bins[1:])/2
    ax.set_xlabel(xlabel)        
    if normed:
        if norm0:
            cnts=cnts/cnts[0]
        elif normval:
            cnts=cnts/normval
        else:
            cnts= cnts/np.max(cnts)   
    if not cumulative:
        ax.step(bncenters, cnts, linewidth=linewidth, linestyle=linestyle, label=label, color=c)
        ylim = [0, np.max(cnts)+np.max(cnts)/10]
        set_aesthetics(ax=ax, fig=fig, makefigax=False, ylim=ylim, xlim=range, xlabel=xlabel, ylabel=ylabel)
        int_ = scipy.integrate.simps(cnts, x=bncenters)
        return {'fig':fig, 'ax':ax,'bncenters':bncenters, 'cnts':cnts, 'int_':int_}
    else:
        if normed:
            cnts = cnts/np.sum(cnts)
        int_ = scipy.integrate.simps(cnts, x=bncenters)
        ylim = [0, np.max(np.cumsum(cnts))+np.max(np.cumsum(cnts))/10]
        set_aesthetics(ax=ax, fig=fig, makefigax=False, ylim=ylim, xlim=range, xlabel=xlabel, ylabel=ylabel, datetime=datetime)        
        if reverse:
            plt.plot(bncenters[::-1], np.cumsum(cnts[::-1]), 'o', linestyle=linestyle, label=label, color=c)
            plt.gca().invert_xaxis()
            return {'fig':fig, 'ax':ax,'bncenters':bncenters[::-1],'cumsum':np.cumsum(cnts[::-1]), 'int_':int_}
        else:        
            plt.plot(bncenters, np.cumsum(cnts), label=label, color=c)
            return {'fig':fig, 'ax':ax,'bncenters':bncenters, 'cumsum':np.cumsum(cnts), 'int':int_}
def plot3d(x,y,z, ax = None, fig = None, makefigax=False):

    from mpl_toolkits.mplot3d import Axes3D
    fig, ax  = figure_check(fig=fig, ax=ax, makefig=makefigax, projection='3d')
    ax.scatter(x,y,z, s=0.1, alpha=0.1)
    plt.show()
    return {'fig':fig,'ax':ax}
def plotbar(x,height, fig=None, ax=None, makefigax=False,label='',
            width=7, color='k',xlabel='', ylabel='', title='', xlim=[], ylim=[]):
    
    fig, ax  = figure_check(fig=fig, ax=ax, makefig=makefigax)    
    plt.bar(x,height, width=width, color=color, label='')
    set_aesthetics(fig=fig, ax=ax, xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=ylim)
    return {'fig':fig,'ax':ax}

class Plot:
    def __init__(self, plotfn, fig=None, ax=None):
        self.plotfn = plotfn
        self.fig = fig
        self.axes = ax
    def savefig(self, filename, format='png', bbox_inches='tight'):
        plt.savefig(filename, dpi=250, format=format, bbox_inches=bbox_inches)
    def close(self):
        plt.close()
        
    
    def plot(self, *args, **kwargs):
        plotoutputs = self.plotfn(*args, **kwargs)            
        print(plotoutputs)                        
        return plotoutputs
        
        
hist1d = Plot(plothist)              
hist2d = Plot(plot2dhist)                
scat = Plot(scatter)
p3d  = Plot(plot3d)
bar = Plot(plotbar)



