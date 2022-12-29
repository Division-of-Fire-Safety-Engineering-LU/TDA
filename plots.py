"""
Arthur Rohaert
Thu Jun 30 10:33:47 2022
"""

# %% ------------------------------------------------------ import the packages
###############################################################################

import os
import numpy as np
import matplotlib.pyplot as plt

# %% ------------------------------------------------------ get number of lanes
###############################################################################

def get_number_of_lanes(df):
    nol = 0
    while True:
        column_name = 'L' + str(nol+1) + ' Speed (km/h)'
        if column_name not in df:
            break
        if df[column_name].isnull().all():
            break
        nol += 1
    return nol

# %% --------------------------- plot relations between density, speed and flow
###############################################################################

def plot_detector_diagram(title, df, input_information):
# --------------------------------------------- set up lanes and miles settings
    nol = get_number_of_lanes(df)
    lanes = [f'L{i+1}' for i in range(nol)]
    labels = [f'Lane {i+1}' for i in range(nol)]
    labels[0] = labels[0] + ' (fast lane)'
    labels[-1] = labels[-1] + ' (slow lane)'
    colours = {'T'  : (0.75, 0, 0),        # red
               'L1' : (0.75, 0, 0),        # red
               'L2' : (0.15, 0.25, 0.50),  # blue
               'L3' : (0.40, 0.60, 0.25),  # green
               'L4' : (1, 0.75, 0),        # yellow
               'L5' : 'tab:purple',
               'L6' : 'tab:brown',
               'L7' : 'tab:pink',
               'L8' : 'tab:gray',
               'L9' : 'tab:olive',
               'L10': 'tab:cyan',
               'L11': 'tab:blue',
               'L12': 'tab:orange',
               'L13': 'tab:green',
               'L14': 'tab:red'}

# --------------------------------------------------------------- set up canvas
    # create figure
    c = 0.393701 # cm per inch
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24*c, 12*c))
    plt.rcParams["font.family"] = "Times New Roman"
    fig.suptitle(title)
    fig.delaxes(ax4)

    # show grid on all figures
    for ax in fig.get_axes():
        ax.grid(True)
        ax.tick_params(axis='both', which='both',
                       direction = 'in', color=[0,0,0,0])

# ------------------------------------------------------------------- plot data
    for lane in lanes[0:nol]:
        # first plot: speed(density)
        ax1.plot(df[lane +' Density (veh/km/lane)'],
                 df[lane + ' Speed (km/h)'],
                 '.', markersize=1, color = colours[lane])
        # second plot: speed(flow)
        ax2.plot(df[lane +' Flow (veh/h/lane)'],
                 df[lane + ' Speed (km/h)'],
                 '.', markersize=1, color = colours[lane])
        # third plot: flow(density)
        ax3.plot(df[lane +' Density (veh/km/lane)'],
                 df[lane + ' Flow (veh/h/lane)'],
                 '.', markersize=1, color = colours[lane])
    ax1.legend(labels, fontsize = 'small', edgecolor = 'black', framealpha = 1)

# ---------------------------------------------------------------- primary axes
    md = input_information["plot_range"][0]
    ms = input_information["plot_range"][1]
    mf = input_information["plot_range"][2]
    
    # density
    ax1.xaxis.set_ticks(np.arange(0, md+1, 20))
    ax3.xaxis.set_ticks(np.arange(0, md+1, 20))
    ax1.axes.xaxis.set_ticklabels([])
    ax3.set_xlabel('Density (veh/km/lane)')
    ax1.set_xlim([0,md])
    ax3.set_xlim([0,md])

    # speed
    ax1.yaxis.set_ticks(np.arange(0, ms+1, 25))
    ax2.yaxis.set_ticks(np.arange(0, ms+1, 25))
    ax2.axes.yaxis.set_ticklabels([])
    ax1.set_ylabel('Speed (km/h)')
    ax1.set_ylim([0,ms])
    ax2.set_ylim([0,ms])

    # flow
    ax2.xaxis.set_ticks(np.arange(0, mf+1, 500))
    ax3.yaxis.set_ticks(np.arange(0, mf+1, 500))
    ax2.set_xlabel('Flow (veh/h/lane)')
    ax3.set_ylabel('Flow (veh/h/lane)')
    ax2.set_xlim([0,mf])
    ax3.set_ylim([0,mf])

# ----------------------------------------------------------------- save figure
    plt.tight_layout()    
    formats = input_information["format"]
    wd = input_information["directory"]
    for form in formats:
        path = os.path.join(wd, 'detector_plots', title+'_FD.'+form)
        if not os.path.isdir(os.path.join(wd,"detector_plots")):
            os.mkdir(os.path.join(wd,"detector_plots"))
        fig.savefig(path, format=form, dpi = 750)
    plt.close(fig)

# %% ------------------------------ plot time series of density, speed and flow
###############################################################################

def plot_detector_series(title, df, input_information):

# --------------------------------------------- set up lanes and miles settings
    nol = get_number_of_lanes(df)
    lanes = [f'L{i+1}' for i in range(nol)]
    labels = [f'Lane {i+1}' for i in range(nol)]
    labels[0] = labels[0] + ' (fast lane)'
    labels[-1] = labels[-1] + ' (slow lane)'
    colours = {'T'  : (0.75, 0, 0),        # red
               'L1' : (0.75, 0, 0),        # red
               'L2' : (0.15, 0.25, 0.50),  # blue
               'L3' : (0.40, 0.60, 0.25),  # green
               'L4' : (1, 0.75, 0),        # yellow
               'L5' : 'tab:purple',
               'L6' : 'tab:brown',
               'L7' : 'tab:pink',
               'L8' : 'tab:gray',
               'L9' : 'tab:olive',
               'L10': 'tab:cyan',
               'L11': 'tab:blue',
               'L12': 'tab:orange',
               'L13': 'tab:green',
               'L14': 'tab:red'}

# --------------------------------------------------------------- set up canvas
    c = 0.393701 # cm to inch
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24*c, 24*c))
    plt.rcParams["font.family"] = "Times New Roman"
    fig.suptitle(title)

# ------------------------------------------------------------------- plot data
    for lane in lanes[0:nol]:
        df[lane +' Density (veh/km/lane)'].plot(ax=ax1,color = colours[lane])
        df[lane + ' Speed (km/h)'].plot(ax=ax2, color = colours[lane])
        df[lane +' Flow (veh/h/lane)'].plot(ax=ax3, color = colours[lane])
    ax1.legend(labels, fontsize = 'small', edgecolor = 'black', framealpha = 1)
    
# ---------------------------------------------------------------- primary axes
    # horizontal axes
    for ax in fig.get_axes():
        ax.grid(True, which="both")
        ax.set_xlabel(None)
        ax.tick_params(axis='both',which='both',direction='in',color=[0,0,0,0])
        indx = df.index.get_level_values('Time')
        ax.set_xlim([min(indx).floor('H'),max(indx).ceil('H')])

    for ax in (ax1, ax2):
        ax.axes.xaxis.set_ticklabels([], minor = True)
        ax.axes.xaxis.set_ticklabels([], minor = False)

    # vertical axes
    md = input_information["plot_range"][0]
    ms = input_information["plot_range"][1]
    mf = input_information["plot_range"][2]
    ax1.set_ylabel('Density (veh/km/lane)')
    ax1.set_ylim([0,md])
    ax1.yaxis.set_ticks(np.arange(0, md+1, 20))
    ax2.set_ylabel('Speed (km/h)')
    ax2.set_ylim([0,ms])
    ax2.yaxis.set_ticks(np.arange(0, ms+1, 25))
    ax3.set_ylabel('Flow (veh/h/lane)')
    ax3.set_ylim([0,mf])
    ax3.yaxis.set_ticks(np.arange(0, mf+1, 500))

# ----------------------------------------------------------------- save figure
    plt.tight_layout()
    formats = input_information["format"]
    wd = input_information["directory"]
    for form in formats:
        path = os.path.join(wd, 'detector_plots', title+'_S.'+form)
        if not os.path.isdir(os.path.join(wd,"detector_plots")):
            os.mkdir(os.path.join(wd,"detector_plots"))
        fig.savefig(path, format=form, dpi = 750)
    plt.close(fig)

# %% ------------------------- plot different data sets and or different models
###############################################################################

def plot_data(input_information, df = None, model = None, title = None):

# --------------------------------------------------------------- set up canvas
    c = 0.393701 # cm per inch
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(19*c, 19/2*c))
    plt.rcParams["font.family"] = "Times New Roman"
    SMALL_SIZE = 8
    LEGEND_SIZE = 7
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize

    # show grid on all figures
    for ax in (ax1, ax2, ax3):
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='both',
                       direction = 'in', color=[0,0,0,0])

# ------------------------------------------------------- reference plot styles    
    ref_data_plotstyle = dict(marker='.', markersize=1, ls='')
    ref_model_plotstyle = dict(marker='', lw=1, c='k')
    colours = [(0.40, 0.60, 0.25),(1, 0.75, 0), 'tab:purple','tab:brown',
               'tab:pink','tab:gray','tab:olive', 'tab:cyan','tab:orange',
               'tab:green']
    line_styles = [(0, (1, 1)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)),
                   (0, (3, 10, 1, 10, 1, 10)), (0, (1, 10)),
                   (0, (3, 10, 1, 10)), (0, (5, 10)), (0, (3, 1, 1, 1)),
                   (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]
    
# ------------------------------------------------------------ plot data points
    scenarios = set(df.index.get_level_values(2))
    labels = []
    for scenario in scenarios:
        # select plot style
        data_plotstyle = ref_data_plotstyle
        data_plotstyle['label'] = scenario
        if scenario == 'Evacuation':
            data_plotstyle['color'] = (0.75, 0, 0)
            data_plotstyle['zorder'] = 2
        elif scenario == 'Routine':
            data_plotstyle['color'] = (0.15, 0.25, 0.50)
            data_plotstyle['zorder'] = 1
        else:
            data_plotstyle['c'] = colours.pop(0)
            data_plotstyle['zorder'] = 1

        # plot data
        ax1.plot(df.loc[:,:,scenario]['T Density (veh/km/lane)'],
                 df.loc[:,:,scenario]['T Speed (km/h)'],**data_plotstyle)
        ax2.plot(df.loc[:,:,scenario]['T Flow (veh/h/lane)'],
                 df.loc[:,:,scenario]['T Speed (km/h)'], **data_plotstyle)
        ax3.plot(df.loc[:,:,scenario]['T Density (veh/km/lane)'],
                 df.loc[:,:,scenario]['T Flow (veh/h/lane)'], **data_plotstyle)
        ax4.plot(0, 0,  **data_plotstyle) # for the legend
        labels.append(f"{scenario.capitalize()} data")

# ----------------------------------------------------------------- plot models
    for scenario in scenarios:
        # select plot style
        model_plotstyle = ref_model_plotstyle
        if scenario == 'Evacuation':
            model_plotstyle['ls'] = '-'
            model_plotstyle['zorder'] = 4
        elif scenario == 'Routine':
            model_plotstyle['ls'] = '--'
            model_plotstyle['zorder'] = 3
        else:
            model_plotstyle['ls'] = line_styles.pop(0)
            model_plotstyle['zorder'] = 3
            
        if model is not None:
            # calculate model
            parameters = np.array(list(model['Result ('+scenario.lower()+')'].params.valuesdict().values()))
            k_model = np.append(np.linspace(0,99,100), np.logspace(2,3,100))
            if 'keypoints' in model.keys():
                k_model = np.append(k_model, parameters[model['keypoints']])
                k_model.sort()
            v_model = model['function'](k_model, *parameters)
            q_model = k_model * v_model

            # plot model
            ax1.plot(k_model, v_model, **model_plotstyle)
            ax2.plot(q_model, v_model, **model_plotstyle)
            ax3.plot(k_model, q_model, **model_plotstyle)
            ax4.plot(0,0, **model_plotstyle) # for the legend
            labels.append(f"{model['name']} fit to {scenario.lower()} data")

# ---------------------------------------------------------------- primary axes
    md = input_information["plot_range"][0]
    ms = input_information["plot_range"][1]
    mf = input_information["plot_range"][2]
    # density
    for ax in (ax1, ax3):
        ax.xaxis.set_ticks(np.arange(0, md+1, 20))
        ax.set_xlim([0,md])
    ax1.axes.xaxis.set_ticklabels([])
    ax3.set_xlabel('Density (veh/km/lane)')

    # speed
    for ax in (ax1, ax2):
        ax.yaxis.set_ticks(np.arange(0, ms+1, 25))
        ax.set_ylim([0,ms])
    ax2.axes.yaxis.set_ticklabels([])
    ax1.set_ylabel('Speed (km/h)')

    # flow
    ax2.xaxis.set_ticks(np.arange(0, mf+1, 500))
    ax3.yaxis.set_ticks(np.arange(0, mf+1, 500))
    ax2.set_xlabel('Flow (veh/h/lane)')
    ax3.set_ylabel('Flow (veh/h/lane)')
    ax2.set_xlim([0,mf])
    ax3.set_ylim([0,mf])

    # invisible figure (which is there for the legend)
    ax4.axes.xaxis.set_ticklabels([])
    ax4.axes.yaxis.set_ticklabels([])
    ax4.tick_params(axis='both',which='both',direction='in',color=[0,0,0,0])
    ax4.patch.set_alpha(0)
    for edge in ('bottom','top','right','left'):
        ax4.spines[edge].set_color([0,0,0,0])
    ax4.set_xlim([1,2])

# ----------------------------------------------------------------- plot legend
    if len(labels) < 3:
        ax1.legend(labels,
                   edgecolor = 'black',
                   framealpha = 1,
                   fancybox='round')
    else:
        ax4.legend(labels,
                   loc = 'lower center',
                   mode = 'expand',
                   bbox_to_anchor = (0,0,1,0.76),
                   edgecolor = 'black')

# ----------------------------------------------------------------- save figure
    if not title:
        title = ' '.join([str(elem) for elem in labels])
        if len(title) > 100:
            title = title[:100]+'...'

    plt.tight_layout()
    formats = input_information["format"]
    wd = input_information["directory"]
    for form in formats:
        path = os.path.join(wd, title+'.'+form)
        fig.savefig(path, format=form, dpi = 750)
    plt.close(fig)
    