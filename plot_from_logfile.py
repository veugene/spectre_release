description = "Plot the contents of text training logs."

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import argparse
import seaborn
import numpy as np

def parse():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--max_x',
                        help="maximum x value on the plot",
                        required=True, type=int)
    parser.add_argument('--max_y',
                        help="maximum y value on the plot",
                        required=False, default='auto')
    parser.add_argument('--keys',
                        help="the metric keys to scrub from the logs and plot",
                        required=True, nargs='+', type=str)
    parser.add_argument('--separate_keys',
                        help="whether to create a separate plot for each key",
                        required=False, default=False, action='store_true')
    parser.add_argument('--title',
                        help="title of the plot",
                        required=True, type=str)
    parser.add_argument('--source',
                        help="(list of) folder(s) containing training logs"
                        "to plot",
                        required=True, nargs='+', type=str)
    parser.add_argument('--experiment_ID',
                        help="(list of) experiment ID(s), one for each source",
                        required=False, nargs='+', type=str)
    parser.add_argument('--dest',
                        help="write the plot to this file",
                        required=False, default="plot.png", type=str)
    return parser.parse_args()
    
    
def scrub(path, keys):
    history = OrderedDict()
    for key in keys:
        history[key] = []
        
    with open(path, 'rt') as f:
        for l in f:
            for key in keys:
                if key in l:
                    history[key].append( float(l.split('=')[-1]) )
    return history
        

if __name__=='__main__':
    # Get all arguments
    args = parse()
    
    # Verify max_y is correct
    if args.max_y != 'auto' \
                  and not isinstance(args.max_y, int) and float(args.max_y)<=0:
        raise ValueError("max_y is invalid : {}".format(args.max_y))

    # Convert all source urls to absolute, relative to the current working dir
    cwd = os.getcwd()
    source_list = []
    for source in args.source:
        if os.path.isabs(source):
            source_list.append(source)
        else:
            source_list.append(os.path.join(cwd, source))
    
    # Verify that each source directory contains a training_log.txt
    for source in source_list:
        bad_source = []
        if not os.path.exists(os.path.join(source, "training_log.txt")):
            bad_source.append(source)
        if len(bad_source)>0:
            raise Exception("the following source directories do"+\
                            "not contain a `training_log.txt`:"+\
                            "".join(["\n{}"]*len(bad_source))
                            .format(*bad_source))
        
    # Assign IDs to sources
    ID_list = []
    for source in source_list:
        ID_list.append(source.rsplit('/')[-1])
    if args.experiment_ID is not None:
        for i, ID in enumerate(args.experiment_ID):
            ID_list[i] = ID
    
    # Scrub history files
    history = OrderedDict()
    for source, ID in zip(source_list, ID_list):
        history[ID] = scrub(os.path.join(source, "training_log.txt"),
                            args.keys)
            
    # Color generator for the plots
    def gen_colors(num_colors):
        for c in seaborn.color_palette('hls', n_colors=num_colors):
            yield c
    
    # Plot
    if args.separate_keys and len(args.keys)>1:
        fig, ax_list = plt.subplots(nrows=len(args.keys)//2+1,
                                    ncols=len(args.keys)%2+1)
    else:
        fig, ax_list = plt.subplots(nrows=1, ncols=1)
    if not hasattr(ax_list, '__len__'):
        ax_list = np.array(ax_list)
    ax_list = ax_list.flatten()
    if not args.separate_keys:
        color_generator = gen_colors(num_colors=len(args.keys)*len(ID_list))
    for i, key in enumerate(args.keys):
        if args.separate_keys:
            ax = ax_list[i]
            color_generator = gen_colors(num_colors=len(ID_list))
            if args.max_y=='auto':
                max_y = max([max(history[ID][key]) for id in ID_list])*1.1
            title = args.title+"  ("+key+")"
        else:
            ax = ax_list[0]
            if args.max_y=='auto':
                max_y = max([max(history[ID][key]) \
                                 for key in history[ID].keys() \
                                 for ID in history.keys()])*1.1
            title = args.title
        if args.max_y!='auto':
            max_y = float(args.max_y)
        
        ax.set_title(title)
        ax.set_xlabel("number of epochs")
        ax.axis([0, args.max_x, 0, max_y])
        
        for ID in ID_list:
            label = ID
            if not args.separate_keys:
                label = key+"__"+label
            ax.plot(history[ID][key][:args.max_x],
                    color=next(color_generator), label=label)
            
    ax_list[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(top=1.5)
    fig.savefig(args.dest, bbox_inches='tight')
