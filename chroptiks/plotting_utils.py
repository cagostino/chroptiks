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
plt.rc('text', usetex=False)#set to true if available?
plt.ion()

dtFmt=mdates.DateFormatter('%b-%Y')

def finite(arr):    
    '''
    Function Description:
        This function is used to check if an array is finite.
    Args:
        arr = np.array
    Unit Test:
        >>>len(finite(np.random.rand(100)))
        100
        >>>len(finite([1,2,3,4,5, np.nan]))
        5
        
    
    '''
    fin = np.isfinite(arr) 
    return fin
def filter_copy(arr, filter):
    '''np.random.rand(10
    Function Description:
        This function is used to filter an array.
    Args:

        arr = np.array
        filter = np.array
    Unit Test:
        >>>len(filter_copy(np.arange(100), np.arange(100)%2==0))
        50  
    '''
    arr_copy = np.copy(arr[filter])
    return arr_copy
def get_plotting_bounds(quant, upper = 0.99, lower=0.01):
    '''
    Function Description:
        This function is used to get the plotting bounds of an array.
    Args:
        quant = np.array
        upper = float
        lower = float
    Unit Test:
        >>>get_plotting_bounds(np.random.rand(100))
        [0.0, 1.0]
        >>>get_plotting_bounds([1,2,3,4,5,6,7,8,9,10])
        [1, 10]
    '''
    return [np.sort(quant)[int(lower*len(quant))], np.sort(quant)[int(upper*len(quant))]]
def get_within_bounds(quant, mn,mx):
    '''
    Function Description:
        This function is used to get the values within a certain range.
    Args:
        quant = np.array
        mn = float
        mx = float
    Unit Test:
        >>>filter = get_within_bounds(np.arange(100), 30,40)
        >>>len(filter)
        10
    '''
    return (quant<mx) & (quant>mn)




def makefig(shape=(1,1), projection=None, sharex=None, sharey=None, **kwargs):
    '''
    Function Description:
        This function is used to create a figure with subplots. It is a wrapper for plt.figure() and fig.add_subplot() that allows for the creation of multiple subplots in a single figure.
    
    Args:
        shape = tuple like (1,1) (2,1) (1,2) etc.    
        projection = None or str
        sharex = None or bool
        sharey = None or bool
        **kwargs = additional keyword arguments for plt.figure()
    Unit Test:

        >>>fig, ax = makefig(shape=(1,1), projection=None, sharex=None, sharey=None)
        
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
    '''
    Function Description:
        This function is used to check if a figure and axis exist. If they do not exist, it creates them.   
    Args:
        fig = None or plt.figure
        ax = None or plt.axis
        shape = tuple like (1,1) (2,1) (1,2) etc.    
        projection = None or str
        sharex = None or bool
        sharey = None or bool
        makefigax = bool
        **kwargs = additional keyword arguments for plt.figure()
    Unit Test:
    
        >>>fig, ax = figure_check(fig=None, ax=None, shape=(1,1), projection=None, sharex=None, sharey=None, makefigax=True)
        
    '''
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
    '''
    Function Description:
        This function is used to set the aesthetics of a plot. It is a wrapper for ax.set_xlabel(), ax.set_ylabel(), ax.set_aspect(), ax.set_xlim(), ax.set_ylim(), ax.set_title(), ax.xaxis.set_major_formatter(), ax.set_xticklabels(), plt.minorticks_on()
    Args:
        fig = None or plt.figure
        ax = None or plt.axis
        makefigax = bool
        **kwargs = additional keyword arguments for plt.figure()
    Unit Test:
    set_aesthetics
    >>>fig, ax = figure_check(fig=None, ax=None, shape=(1,1), projection=None, sharex=None, sharey=None, makefigax=True)
    >>>set_aesthetics(fig=fig, ax=ax, makefigax=False, xlim=[0,1], ylim=[0,1], xlabel='X', ylabel='Y', aspect='auto', title='Title', datetime=True)    
    '''
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
    return fig,ax

            
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
    '''
    Function Description:
        This function is used to create a scatter plot. It is a wrapper for plt.scatter() and plot2dhist().
    Args:
        x = np.array
        y = np.array
        plotdata = bool
        fig = None or plt.figure
        ax = None or plt.axis
        makefigax = bool
        facecolor = None or str
        cmap = str
        edgecolor = str
        alpha = float
        color = str
        marker = str
        s = float
        vmin = float
        vmax = float
        label = str
        ccode = np.array
        binlabel = str
        bin_y = bool
        lim = bool
        setplotlims = bool
        bin_stat_y = str
        size_y_bin = float
        counting_thresh = int
        percentiles = bool
        linecolor = str
        linewidth = float
        xlabel = str
        ylabel = str

    Unit Test:
    scatter
        >>>fig, ax = figure_check(fig=None, ax=None, shape=(1,1), projection=None, sharex=None, sharey=None, makefigax=True)
        >>>scatter(x=np.random.rand(100), y=np.random.rand(100), plotdata=True,  fig=fig, ax=ax, makefigax=False)
    
    '''
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
    
    '''
    Function Description:
        This function is used to create a scatter plot with error bars. It is a wrapper for plt.errorbar() and plot2dhist().
    Args:
        x = np.array
        y = np.array
        plotdata = bool
        fig = None or plt.figure
        ax = None or plt.axis
        makefigax = bool
        xerr = np.array
        yerr = np.array
        elinewidth = float
        capsize = float
        capthick = float
        ecolor = str
        fmt = str
        label = str
        binlabel = str
        bin_y = bool
        lim = bool
        setplotlims = bool
        bin_stat_y = str
        size_y_bin = float
        counting_thresh = int
        percentiles = bool
        linecolor = str
        linewidth = float
        xlabel = str
        ylabel = str
        aspect = str
        xlim = np.array
        ylim = np.array
    Unit Test:

        >>>fig, ax = figure_check(fig=None, ax=None, shape=(1,1), projection=None, sharex=None, sharey=None, makefigax=True)
        >>>error(x=np.random.rand(100), y=np.random.rand(100), plotdata=True,  fig=fig, ax=ax, makefigax=False)
        
        '''
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
               percentile_levels = [16, 84],
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
    '''
    Function Description:
        This function is used to create a 2D histogram. It is a wrapper for plt.imshow() and plt.plot().
    Args:

        x = np.array
        y = np.array
        nx = int
        ny = int
        bin_y = bool
        bin_stat_y = str
        ybincolsty = str
        plotlines = bool
        ybincolsty_perc = str
        nbins = int
        size_y_bin = float
        bin_quantity = np.array
        percentiles = bool
        percentile_levels = list
        counting_thresh = int
        linewid = float
        ccode = np.array
        ccode_stat = np.nanmedian
        ccodename = str
        ccodelim = list
        ccode_bin_min = int
        cmap = str
        show_cbar = bool
        dens_scale = float
        label = str
        zorder = int
        nan = bool
        data = bool
        fig = None or plt.figure
        makefigax = bool
        ax = None or plt.axis
        xlim = np.array
        ylim = np.array
        xlabel = str
        ylabel = str
        lim = bool
        setplotlims = bool
        aspect = str
    Unit Test:
        >>>fig, ax = figure_check(fig=None, ax=None, shape=(1,1), projection=None, sharex=None, sharey=None, makefigax=True)
        >>>plot2dhist(x=np.random.rand(100), y=np.random.rand(100), nx=200, ny=200, bin_y=False, bin_stat_y='mean', ybincolsty='r-', plotlines=True, ybincolsty_perc='r-.', nbins=25, size_y_bin=0, bin_quantity=[], percentiles=False, percentile_levels=[16, 84], counting_thresh=20, linewid=2, ccode=[], ccode_stat=np.nanmedian, ccodename='', ccodelim=[], ccode_bin_min=20, cmap='plasma', show_cbar=True, dens_scale=0.3, label='', zorder=10, nan=True, data=True, fig=fig, makefigax=False, ax=ax, xlim=[0,0], ylim=[0,0], xlabel='', ylabel='', lim=False, setplotlims=False, aspect='auto')
        
        
    
    '''
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
                bins_up = []
                bins_down = []
                for i in range(len(xedges)-1):
                    binned_ = np.where((x>xedges[i])&(x<xedges[i+1]))[0]
                    if i in good_y:
                        bins_up.append(np.percentile(y[binned_], 16))
                        bins_down.append(np.percentile(y[binned_], 84))
                    else:
                        bins_down.append(np.nan)
                        bins_up.append(np.nan)                        
                bins_up = np.array(bins_up)
                bins_down = np.array(bins_down)
            if len(bin_quantity) !=0:
                avg_quant, _xedges, _binnum = scipy.stats.binned_statistic(x,bin_quantity, statistic=bin_stat_y, bins = nbins,range=xlim)
                avg_quant = avg_quant[good_y]
            else:
                avg_quant = bin_quantity
            if ax:
                if plotlines:
                    ax.plot(xmid[good_y], avg_y[good_y], ybincolsty, linewidth=linewid, label=label, zorder=zorder)
                    if percentiles:
                        ax.plot(xmid[good_y], bins_up[good_y], ybincolsty_perc, linewidth=linewid, label=label, zorder=zorder)
                        ax.plot(xmid[good_y], bins_down[good_y], ybincolsty_perc, linewidth=linewid, label=label, zorder=zorder)
                if percentiles:
                    return {'fig':fig, 'ax':ax, 'xmid':xmid[good_y], 'avg_y':avg_y[good_y], 'avg_quant':avg_quant, 'bins_down':bins_down[good_y], 'bins_up':bins_up[good_y]}
                return {'fig':fig,'ax': ax, 'xmid':xmid[good_y], 'avg_y':avg_y[good_y],'avg_quant': avg_quant}
            else:
                if plotlines:
                    plt.plot(xmid[good_y], avg_y[good_y],ybincolsty, linewidth=linewid, label=label, zorder=zorder)
                    if percentiles: 
                        plt.plot(xmid[good_y], bins_up[good_y], ybincolsty_perc, linewidth=linewid, label=label, zorder=zorder)
                        plt.plot(xmid[good_y], bins_down[good_y], ybincolsty_perc, linewidth=linewid, label=label, zorder=zorder)
                        
                if percentiles: 
                    return {'fig':fig,'ax':ax,'xmid': xmid[good_y], 'avg_y':avg_y[good_y], 'avg_quant':avg_quant, 'bins_down':bins_down[good_y], 'bins_up':bins_up[good_y]      }            
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
    Function Description:
        This function is used to create a histogram. It is a wrapper for plt.hist() and plt.step().
    Args:
        x = np.array
        bins = int
        range = tuple
        linewidth = float
        datetime = bool
        cumulative = bool
        reverse = bool
        ylim = bool
        label = str
        linestyle = str
        c = str
        density = bool
        xlabel = str
        ylabel = str
        normed = bool
        norm0 = bool
        normval = float
        integrate = bool
        fig = None or plt.figure
        ax = None or plt.axis
        makefigax = bool
    Unit Test:
        >>>fig, ax = figure_check(fig=None, ax=None, shape=(1,1), projection=None, sharex=None, sharey=None, makefigax=True)
        >>>plothist(x=np.random.rand(100), bins=10, range=(), linewidth=1, datetime=False, cumulative=False, reverse=False, ylim=False, label='',linestyle='--', c='k', density=False, xlabel='',ylabel='Counts', normed=False, norm0= False, normval=None, integrate = False,fig=fig, ax=ax, makefigax=False)
        
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
    '''
    Function Description:
        This function is used to create a 3D plot. It is a wrapper for plt.scatter() and plt.show().
    Args:
        x = np.array
        y = np.array
        z = np.array
        ax = None or plt.axis
        fig = None or plt.figure
        makefigax = bool
    Unit Test:
        >>>fig, ax = figure_check(fig=None, ax=None, shape=(1,1), projection=None, sharex=None, sharey=None, makefigax=True)
        >>>plot3d(x=np.random.rand(100), y=np.random.rand(100), z=np.random.rand(100), ax=ax, fig=fig, makefigax=False)
        
    '''
    from mpl_toolkits.mplot3d import Axes3D
    fig, ax  = figure_check(fig=fig, ax=ax, makefig=makefigax, projection='3d')
    ax.scatter(x,y,z, s=0.1, alpha=0.1)
    plt.show()
    return {'fig':fig,'ax':ax}
def plotbar(x,height, fig=None, ax=None, makefigax=False,label='',
            width=7, color='k',xlabel='', ylabel='', title='', xlim=[], ylim=[]):
    '''
    Function Description:
        This function is used to create a bar plot. It is a wrapper for plt.bar().
    Args:
        x = np.array
        height = np.array
        fig = None or plt.figure
        ax = None or plt.axis
        makefigax = bool
        label = str
        width = float
        color = str
        xlabel = str
        ylabel = str
        title = str
        xlim = np.array
        ylim = np.array
    Unit Test:
        >>>fig, ax = figure_check(fig=None, ax=None, shape=(1,1), projection=None, sharex=None, sharey=None, makefigax=True)
        >>>plotbar(x=np.random.rand(100), height=np.random.rand(100), fig=fig, ax=ax, makefigax=False,label='', width=7, color='k',xlabel='', ylabel='', title='', xlim=[], ylim=[])
        
    '''
    
    fig, ax  = figure_check(fig=fig, ax=ax, makefig=makefigax)    
    plt.bar(x,height, width=width, color=color, label='')
    set_aesthetics(fig=fig, ax=ax, xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=ylim)
    return {'fig':fig,'ax':ax}

class Plot:
    '''
    A wrapper class for the plotting functions.
    '''
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
        return plotoutputs
        
        
hist1d = Plot(plothist)              
hist2d = Plot(plot2dhist)                
scat = Plot(scatter)
p3d  = Plot(plot3d)
bar = Plot(plotbar)



