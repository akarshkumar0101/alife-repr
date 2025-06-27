import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from cycler import cycler

def set_plot_style():
    plt.style.use('default')
    # plt.style.use('seaborn-v0_8')

    # ----- change figure size and dpi -----
    plt.rcParams['figure.figsize'] = (8.5, 4)
    plt.rcParams['figure.dpi'] = 150

    # ----- change grid lines to white -----
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = 'white'
    plt.rcParams['grid.linewidth'] = 1.

    # ----- change axes face and edge colors -----
    plt.rcParams['axes.facecolor'] = '#ededed'
    plt.rcParams['axes.facecolor'] = '#e6e6e6'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 1

    # plt.rcParams['font.size'] = 35
    # plt.rcParams['axes.titlesize'] = 35
    # plt.rcParams['axes.labelsize'] = 25
    # plt.rcParams['ytick.labelsize'] = 15
    # plt.rcParams['xtick.labelsize'] = 15
    # plt.rcParams['legend.fontsize'] = 15

    # ----- change font to Go-----
    font_files = font_manager.findSystemFonts(fontpaths=[os.path.expanduser('~/fonts/')])
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
    plt.rcParams['font.family'] = 'Go'
    plt.rcParams['font.sans-serif'] = ['Go']
    plt.rcParams['font.serif'] = ['Go']
    plt.rcParams['font.monospace'] = ['Go Mono']

    # ----- remove tick marks (bc of grid) -----
    plt.rcParams['ytick.major.size'] = 0
    plt.rcParams['xtick.major.size'] = 0
    plt.rcParams['ytick.minor.size'] = 0
    plt.rcParams['xtick.minor.size'] = 0

    # ----- change line width and marker size -----
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 2

    # ----- change legend frame and edge color -----
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.edgecolor'] = 'black'
    plt.rcParams['legend.facecolor'] = '#f0f0f0'

    # ----- change colormap -----
    if 'ak0' in plt.colormaps():
        plt.colormaps.unregister('ak0'); plt.colormaps.unregister('ak0_r')
        plt.colormaps.unregister('ak1'); plt.colormaps.unregister('ak1_r')
    colors = [f"#{c}" for c in '001219-005f73-0a9396-94d2bd-e9d8a6-ee9b00-ca6702-bb3e03-9b2226'.split('-')]
    plt.colormaps.register(cmap=matplotlib.colors.LinearSegmentedColormap.from_list('ak0', colors, N=1000))
    plt.colormaps.register(cmap=matplotlib.colors.LinearSegmentedColormap.from_list('ak0_r', colors[::-1], N=1000))
    colors = [f"#{c}" for c in 'b7094c-a01a58-892b64-723c70-5c4d7d-455e89-2e6f95-1780a1-0091ad'.split('-')]
    plt.colormaps.register(cmap=matplotlib.colors.LinearSegmentedColormap.from_list('ak1', colors, N=1000))
    plt.colormaps.register(cmap=matplotlib.colors.LinearSegmentedColormap.from_list('ak1_r', colors[::-1], N=1000))
    plt.rcParams['image.cmap'] = 'ak0'

    # ----- change color cycle -----
    # colors = [f"#{c}" for c in '355070-6d597a-b56576-e56b6f-eaac8b'.split('-')]
    # colors = [f"#{c}" for c in '264653-2a9d8f-e9c46a-f4a261-e76f51'.split('-')]
    # colors = [f"#{c}" for c in '669900-99cc33-ccee66-006699-3399cc-990066-cc3399-ff6600-ff9900-ffcc00'.split('-')]
    colors = [f"#{c}" for c in '001219-005f73-0a9396-94d2bd-e9d8a6-ee9b00-ca6702-bb3e03-ae2012-9b2226'.split('-')]
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)