import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

'''
plot the classification curve along different ratio
'''

plot_choice = 'A+X'  # X or A+X or RN (Recall and Ndcg)
dataset = 'pubmed'  # cora, citeseer, pubmed, steam

if plot_choice == 'X':
    if dataset=='cora':
        NeighAggre = [0.3396,0.4358,0.5145,0.5692,0.6248]
        GCN = [0.455,0.3646,0.3867,0.3886,0.3943]
        VAE = [0.2277,0.2512,0.2661,0.2723,0.2826]
        LFI = [0.2964,0.7228,0.7383,0.7619,0.7644]
    elif dataset=='citeseer':
        NeighAggre = [0.3087,0.3871,0.4433,0.507,0.5539]
        GCN = [0.2717,0.3429,0.3685,0.3587,0.3768]
        VAE = [0.2484,0.2566,0.2535,0.2685,0.2551]
        LFI = [0.4936,0.52,0.5603,0.5946,0.601]
    elif dataset=='pubmed':
        NeighAggre = [0.3852,0.4091,0.4395,0.4869,0.515]
        GCN = [0.3993,0.4012,0.3997,0.4013,0.3992]
        VAE = [0.3761,0.3894,0.3934,0.3951,0.3991]
        LFI = [0.4423,0.4463,0.4502,0.4676,0.4652]
    else:
        raise Exception

    # plot loss curve
    font = {'family': 'Times New Roman',
            'color': 'black',
            'weight': 'bold',
            'size': 15,
            }
    mycolor = np.array([[224, 32, 32],
                        [255, 192, 0],
                        [32, 160, 64],
                        [48, 96, 192],
                        [192, 48, 192]]) / 255.0
    mymarker = ['1', '2', 's', '*', 'H', 'D', 'o', '>']

    my_line_width = 3
    my_marker_size = 10

    # plot train G/D curve
    plt.figure(1)
    plt.style.use('ggplot')
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    x_axix = range(len(NeighAggre))
    plt.plot(x_axix, LFI, color=mycolor[0], label='NANG', marker=mymarker[0], linewidth=my_line_width, markersize=my_marker_size)
    plt.plot(x_axix, NeighAggre, color=mycolor[1], marker=mymarker[1], label='NeighAggre', linewidth=my_line_width, markersize=my_marker_size)
    plt.plot(x_axix, GCN, color=mycolor[2], label='GCN', marker=mymarker[2], linewidth=my_line_width, markersize=my_marker_size)
    plt.plot(x_axix, VAE, color=mycolor[3], label='VAE', marker=mymarker[3], linewidth=my_line_width, markersize=my_marker_size)

    my_legend = plt.legend(loc='best', fontsize=12)
    frame = my_legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')

    plt.xticks(x_axix, [0.08, 0.16, 0.24, 0.32, 0.40])
    plt.tick_params(labelsize='10')
    plt.xlabel('Train ratio', fontdict=font)
    plt.ylabel('Accuracy', fontdict=font)
    # plt.show()
    plt.savefig(os.path.join(os.getcwd(), 'figures', 'LFI', '{}_X_fts_spaisity_acc.png'.format(dataset)))

elif plot_choice == 'A+X':
    if dataset=='cora':
        NeighAggre = [0.3601,0.4555,0.5372,0.594,0.6494]
        GCN = [0.5817,0.4131,0.4227,0.4325,0.4387]
        VAE = [0.2285,0.2503,0.267,0.2728,0.3011]
        LFI = [0.3302,0.8158,0.8152,0.8344,0.8327]
        low_bound = [0.7631,0.7631,0.7631,0.7631,0.7631]
    elif dataset=='citeseer':
        NeighAggre = [0.3061,0.3818,0.4263,0.4966,0.5413]
        GCN = [0.3294,0.3831,0.4031,0.3944,0.4079]
        VAE = [0.2488,0.2596,0.2656,0.2707,0.2663]
        LFI = [0.5633,0.6105,0.6352,0.65,0.6599]
        low_bound = [0.5651,0.5651,0.5651,0.5651,0.5651]
    elif dataset=='pubmed':
        NeighAggre = [0.4719,0.5375,0.5862,0.6238,0.6564]
        GCN = [0.4196,0.4206,0.4205,0.42,0.4203]
        VAE = [0.3834,0.3926,0.3967,0.397,0.4007]
        LFI = [0.7006,0.7262,0.7264,0.742,0.7537]
        low_bound = [0.7125,0.7125,0.7125,0.7125,0.7125]
    else:
        raise Exception

    # plot loss curve
    font = {'family': 'Times New Roman',
            'color': 'black',
            'weight': 'bold',
            'size': 15,
            }
    mycolor = np.array([[224, 32, 32],
                        [255, 192, 0],
                        [32, 160, 64],
                        [48, 96, 192],
                        [192, 48, 192]]) / 255.0
    mymarker = ['1', '2', 's', '*', 'H', 'D', 'o', '>']

    my_line_width = 3
    my_marker_size = 10

    # plot train G/D curve
    plt.figure(1)
    plt.style.use('ggplot')
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    x_axix = range(len(NeighAggre))
    plt.plot(x_axix, LFI, color=mycolor[0], label='NANG', marker=mymarker[0], linewidth=my_line_width, markersize=my_marker_size)
    plt.plot(x_axix, NeighAggre, color=mycolor[1], marker=mymarker[1], label='NeighAggre', linewidth=my_line_width, markersize=my_marker_size)
    plt.plot(x_axix, GCN, color=mycolor[2], label='GCN', marker=mymarker[2], linewidth=my_line_width, markersize=my_marker_size)
    plt.plot(x_axix, VAE, color=mycolor[3], label='VAE', marker=mymarker[3], linewidth=my_line_width, markersize=my_marker_size)
    plt.plot(x_axix, low_bound, color=mycolor[4], label='only A', linestyle='--', linewidth=my_line_width, markersize=my_marker_size)

    my_legend = plt.legend(loc='best', fontsize=12)
    frame = my_legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')

    plt.xticks(x_axix, [0.08, 0.16, 0.24, 0.32, 0.40])
    plt.tick_params(labelsize='10')
    plt.xlabel('Train ratio', fontdict=font)
    plt.ylabel('Accuracy', fontdict=font)
    # plt.show()
    plt.savefig(os.path.join(os.getcwd(), 'figures', 'LFI', '{}_AX_fts_spaisity_acc.png'.format(dataset)))

elif plot_choice == 'RN':
    if dataset=='cora':
        R_NeighAggre = [0.054116,0.082986,0.103679,0.125224,0.141333]
        R_GCN = [0.16485,0.162966,0.169446,0.176151,0.178563]
        R_VAE = [0.056841,0.077480,0.094105,0.109308,0.122863]
        R_LFI = [0.165169,0.192841,0.199909,0.217072,0.218244]

        N_NeighAggre = [0.053015,0.084353,0.109625,0.134569,0.154842]
        N_GCN = [0.196131,0.195857,0.200174,0.206071,0.207894]
        N_VAE = [0.064809,0.092743,0.114055,0.132221,0.1452]
        N_LFI = [0.196172,0.226122,0.236231,0.2531,0.254618]
    elif dataset=='citeseer':
        R_NeighAggre = [0.024219,0.041911,0.060889,0.078351,0.090872]
        R_GCN = [0.089447,0.104188,0.106812,0.1005,0.109709]
        R_VAE = [0.021965,0.037979,0.045481,0.058014,0.066876]
        R_LFI = [0.09274,0.104734,0.112903,0.123007,0.128066]

        N_NeighAggre = [0.029395,0.050927,0.074646,0.098405,0.115508]
        N_GCN = [0.123255,0.131361,0.136763,0.137167,0.142371]
        N_VAE = [0.026638,0.043098,0.056304,0.07225,0.083909]
        N_LFI = [0.117306,0.137767,0.150541,0.165027,0.172966]
    elif dataset=='steam':
        R_NeighAggre = [0.065931,0.074626,0.081131,0.084989,0.088158]
        R_GCN = [0.327228,0.325766,0.326122,0.325782,0.325804]
        R_VAE = [0.067843,0.073942,0.077968,0.080204,0.082005]
        R_LFI = [0.35534,0.358121,0.359822,0.357787,0.356031]

        N_NeighAggre = [0.09246,0.104696,0.113246,0.116874,0.120471]
        N_GCN = [0.403146,0.401912,0.40297,0.402745,0.402568]
        N_VAE = [0.094379,0.102701,0.10776,0.110941,0.113366]
        N_LFI = [0.429275,0.431764,0.434909,0.434638,0.433209]
    else:
        raise Exception

    # plot loss curve
    font = {'family': 'Times New Roman',
            'color': 'black',
            'weight': 'bold',
            'size': 15,
            }
    mycolor = np.array([[224, 32, 32],
                        [255, 192, 0],
                        [32, 160, 64],
                        [48, 96, 192],
                        [192, 48, 192]]) / 255.0
    mymarker = ['1', '2', 's', '*', 'H', 'D', 'o', '>']

    my_line_width = 3
    my_marker_size = 10

    # plot recall curve
    plt.figure(1)
    plt.style.use('ggplot')
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    x_axix = range(len(R_NeighAggre))
    plt.plot(x_axix, R_LFI, color=mycolor[0], label='NANG', marker=mymarker[0], linewidth=my_line_width, markersize=my_marker_size)
    plt.plot(x_axix, R_NeighAggre, color=mycolor[1], marker=mymarker[1], label='NeighAggre', linewidth=my_line_width, markersize=my_marker_size)
    plt.plot(x_axix, R_GCN, color=mycolor[2], label='GCN', marker=mymarker[2], linewidth=my_line_width, markersize=my_marker_size)
    plt.plot(x_axix, R_VAE, color=mycolor[3], label='VAE', marker=mymarker[3], linewidth=my_line_width, markersize=my_marker_size)

    my_legend = plt.legend(loc='best', fontsize=12)
    frame = my_legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')

    plt.xticks(x_axix, [0.08, 0.16, 0.24, 0.32, 0.40])
    plt.tick_params(labelsize='10')
    plt.xlabel('Train ratio', fontdict=font)
    if dataset=='steam':
        plt.ylabel('Recall@5', fontdict=font)
    else:
        plt.ylabel('Recall@20', fontdict=font)
    # plt.show()
    plt.savefig(os.path.join(os.getcwd(), 'figures', 'LFI', '{}_fts_spaisity_recall.png'.format(dataset)))

    # plot NDCG curve
    plt.figure(2)
    plt.style.use('ggplot')
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    x_axix = range(len(N_NeighAggre))
    plt.plot(x_axix, N_LFI, color=mycolor[0], label='NANG', marker=mymarker[0], linewidth=my_line_width,
             markersize=my_marker_size)
    plt.plot(x_axix, N_NeighAggre, color=mycolor[1], marker=mymarker[1], label='NeighAggre', linewidth=my_line_width,
             markersize=my_marker_size)
    plt.plot(x_axix, N_GCN, color=mycolor[2], label='GCN', marker=mymarker[2], linewidth=my_line_width,
             markersize=my_marker_size)
    plt.plot(x_axix, N_VAE, color=mycolor[3], label='VAE', marker=mymarker[3], linewidth=my_line_width,
             markersize=my_marker_size)


    my_legend = plt.legend(loc='best', fontsize=12)
    frame = my_legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')

    plt.xticks(x_axix, [0.08, 0.16, 0.24, 0.32, 0.40])
    plt.tick_params(labelsize='10')
    plt.xlabel('Train ratio', fontdict=font)
    if dataset=='steam':
        plt.ylabel('NDCG@5', fontdict=font)
    else:
        plt.ylabel('NDCG@20', fontdict=font)
    # plt.show()
    plt.savefig(os.path.join(os.getcwd(), 'figures', 'LFI', '{}_fts_spaisity_ndcg.png'.format(dataset)))
