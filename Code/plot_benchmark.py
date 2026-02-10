import matplotlib.pyplot as plt
import numpy as np
import os
# import time
import pandas as pd
import numpy as np

import argparse

import parse_data
# from Code import models
# from importlib import reload


"""
Get the values to plot. Helper funciton for cusotm graph making
"""
def get_scores(path_dict):
    dict_1_acc = {}
    dict_5_acc = {}

    for j, key in enumerate(path_dict.keys()):
        read_file = "output.txt"
        file_path = path_dict[key]
        t = file_path
        if t is None:
            print(">", key, " : ", t)
            continue
            # print(">", key, " : ", t)
        path = os.path.join(t, read_file)#  "output.txt"
        try:
            a,b,c = parse_data.read_file_lite(path)
        except FileNotFoundError:
            print(key," : ", t)
            pass 
        
        r=c["res"]
        l_acc_1=[]
        l_acc_5=[]
        for i in r.keys():
            line = r[i]
            acc1, acc5 = parse_data.parses_result(line)
            l_acc_1.append(acc1)
            l_acc_5.append(acc5)
        dict_1_acc[key] = l_acc_1
        dict_5_acc[key] = l_acc_5
    pass
 
    return dict_1_acc, dict_5_acc

"""
Take Classication and plot accuracy. Give it a dictionary of paths. Auto do code. Names will be dumb but should be passed in.
"""
def plot_validation(path_dict, titleName="TITLE HERE"):




    dict_1_acc, dict_5_acc = get_scores(path_dict=path_dict)
    
    fig,axs = plt.subplots()
    dict_1_acc;
    display_dict = {x: dict_1_acc[x][-1] for x in dict_1_acc.keys()}
    display_list=sorted(display_dict, key=display_dict.get)
    axs.bar(display_list, [display_dict[l] for l in display_list], align='edge', width=-0.4, label="Top 1 Acc")
    axs.bar(display_list, [dict_5_acc[l][-1] for l in display_list],  align='edge',width=0.4, label = "Top 5 Acc")
    axs.tick_params(axis='x',rotation=90)
    axs.set_title(titleName)
    return fig,axs

"""
Takse CSV and load in value for ploting
"""
def load_benchmark_csv(data_path):
    df = pd.read_csv(data_path)
    new_label = pd.MultiIndex.from_arrays([ df.iloc[-1], df.columns], names=['Heading', 'SubHeading'])
    df1  = df.set_axis(new_label, axis=1).iloc[0:-1]
    df1;
    cols = df1.columns[df1.dtypes.eq('object')]
    # cosl
    df1[cols] = df1[cols].apply(pd.to_numeric,errors='ignore')
    df1
    return df1

def plot_scatter(data_path, args):
    save_folder = args.save_folder
    df1 = load_benchmark_csv(data_path)
    fig, ax = plt.subplots(figsize=(12,8))

    x=('Aggregate score', 'Batch correction')
    y=('Aggregate score', 'Bio conservation')
    lbl=('Metric Type', "Embedding")

    for index, row in df1[[x,y,lbl]].iterrows():
        # print(str(a))
        batch_score = row[x]
        bio_score = row[y]
        label = row[lbl]
        # print(batch_score, bio_score,label)
        ax.scatter(x=batch_score,y=bio_score,label=label)
        
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_xlabel("Batch Correction Score")
    ax.set_ylabel("Bio Correction Score")
    ax.set_title("Scores")
    fig.tight_layout()

    save_path = os.path.join(save_folder, "Scatter_"+args.file_name + "."+ args.format)
    fig.savefig(save_path, format=args.format)


def plot_benchmark_csv(data_path, args):

    save_folder = args.save_folder
    df1 = load_benchmark_csv(data_path)
    # 'Batch correction', 'Bio conservation', 'Total'
    df2=df1.sort_values(('Aggregate score','Total'), ascending=False)
    df3=df1.sort_values(('Aggregate score','Bio conservation'), ascending=False)
    df4=df1.sort_values(('Aggregate score','Batch correction'), ascending=False)

    ax_tot=df2.plot.bar(x=('Metric Type', "Embedding"), y='Aggregate score', title="All Combined", width=0.8, stacked=False)
    # ax_tot.figure.tight_layout()
    ax_tot.set_title(args.title_name)
    fig_tot = ax_tot.get_figure()
    fig_tot.set_size_inches((10, 8))
    ax_tot.set_ylabel("Score")
    ax_tot.legend(loc='upper right')
    fig_tot.tight_layout()
    save_path = os.path.join(save_folder, "TOT_"+args.file_name + "."+ args.format)
    fig_tot.savefig(save_path, format=args.format)

    ax_bio=df2.plot.bar(x=('Metric Type', "Embedding"), y='Aggregate score', title="All Combined", width=0.8, stacked=False)
    # ax_bio.figure.tight_layout()
    ax_bio.set_title(args.title_name)
    fig_bio = ax_bio.get_figure()
    fig_bio.set_size_inches((10, 8))
    ax_bio.set_ylabel("Score")
    ax_bio.legend(loc='upper right')
    fig_bio.tight_layout()
    # /.figure.savefig("Bio_"+args.file_name)
    save_path = os.path.join(save_folder, "Bio_"+args.file_name + "."+ args.format)
    fig_bio.savefig(save_path, format=args.format)

    ax_batch=df2.plot.bar(x=('Metric Type', "Embedding"), y='Aggregate score', title="All Combined", width=0.8, stacked=False)
    # ax_batch.figure.tight_layout()
    ax_batch.set_title(args.title_name)
    fig_batch = ax_batch.get_figure()
    fig_batch.set_size_inches((10, 8))
    ax_batch.set_ylabel("Score")
    ax_batch.legend(loc='upper right')
    fig_batch.tight_layout()

    save_path = os.path.join(save_folder, "Batch_"+args.file_name + "."+ args.format)
    fig_batch.savefig(save_path, format=args.format)

parser = argparse.ArgumentParser()
parser.add_argument("-data_path",  metavar='DIR', nargs='?', default='./',
                    help = "csv data to check")
parser.add_argument("-file_name",  nargs='?', default='./',
                    help = "What to save file as")
parser.add_argument("-save_folder",  metavar='DIR', nargs='?', default='./',
                    help = "where to save stuff")
parser.add_argument("-title_name",  default='./',
                    help = "Title of Graph")
parser.add_argument("-format",  default='png',
                    help = "save the graph as images (default png)")


if __name__ == "__main__":
    args = parser.parse_args()
    print("Args : ----")
    print(args)
    print("----")
    plot_benchmark_csv(args.data_path, args=args)
    plot_scatter(args.data_path, args=args)

    print("DONE WITH PLOTS")
    pass
