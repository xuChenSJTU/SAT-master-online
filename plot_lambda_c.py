import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

'''
plot the classification curve along different ratio
'''

plot_choice = 'N'  # A+X or R or N (Recall and Ndcg)
dataset = 'cora'  # cora, citeseer, pubmed, steam


if plot_choice == 'X':
    if dataset=='cora':
        LFI = []
        criterion = [0.6248,0.6248,0.6248,0.6248,0.6248,0.6248,0.6248]
    elif dataset=='citeseer':
        LFI = []
        criterion = [0.5539,0.5539,0.5539,0.5539,0.5539,0.5539,0.5539]
    elif dataset=='pubmed':
        LFI = []
        criterion = [0.515,0.515,0.515,0.515,0.515,0.515,0.515]
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
    x_axix = range(len(LFI))
    plt.plot(x_axix, LFI, color=mycolor[0], label='NANG', marker=mymarker[0], linewidth=my_line_width, markersize=my_marker_size)
    plt.plot(x_axix, criterion, color=mycolor[4], label='NeighAggre', linestyle='--', linewidth=my_line_width, markersize=my_marker_size)

    my_legend = plt.legend(loc='best', fontsize=12)
    frame = my_legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')

    plt.xticks(x_axix, [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
    plt.tick_params(labelsize='10')
    plt.xlabel('$\lambda_{c}$', fontdict=font)
    plt.ylabel('Accuracy', fontdict=font)
    # plt.show()
    plt.savefig(os.path.join(os.getcwd(), 'figures', 'LFI', '{}_X_lambda_accuracy.png'.format(dataset)))

elif plot_choice == 'A+X':
    if dataset=='cora':
        LFI = [0.7148,0.7429,0.8006,0.8202,0.8327,0.8397,0.8395]
        criterion = [0.7631,0.7631,0.7631,0.7631,0.7631,0.7631,0.7631]
    elif dataset=='citeseer':
        LFI = [0.4025,0.6082,0.6145,0.6575,0.6599,0.6763,0.6772]
        criterion = [0.5651,0.5651,0.5651,0.5651,0.5651,0.5651,0.5651]
    elif dataset=='pubmed':
        LFI = [0.4208,0.4213,0.4277,0.5369,0.6114,0.7166,0.7537]
        criterion = [0.7125,0.7125,0.7125,0.7125,0.7125,0.7125,0.7125]
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
    x_axix = range(len(LFI))
    plt.plot(x_axix, LFI, color=mycolor[0], label='NANG', marker=mymarker[0], linewidth=my_line_width, markersize=my_marker_size)
    plt.plot(x_axix, criterion, color=mycolor[4], label='only A', linestyle='--', linewidth=my_line_width, markersize=my_marker_size)

    my_legend = plt.legend(loc='best', fontsize=12)
    frame = my_legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')

    plt.xticks(x_axix, [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
    plt.tick_params(labelsize='10')
    plt.xlabel('$\lambda_{c}$', fontdict=font)
    plt.ylabel('Accuracy', fontdict=font)
    # plt.show()
    plt.savefig(os.path.join(os.getcwd(), 'figures', 'LFI', '{}_AX_lambda_accuracy.png'.format(dataset)))

elif plot_choice == 'R':
    # plot for recall
    if dataset=='cora':
        LFI = [0.19787,0.201162,0.206989,0.214487,0.218244,0.22229,0.219685]
        criterion = [0.178563,0.178563,0.178563,0.178563,0.178563,0.178563,0.178563]
    elif dataset=='citeseer':
        LFI = [0.105197,0.120321,0.121163,0.122753,0.128066,0.132498,0.137292]
        criterion = [0.109709,0.109709,0.109709,0.109709,0.109709,0.109709,0.109709]
    elif dataset=='steam':
        LFI = [0.328498,0.348517,0.350193,0.349175,0.356031,0.348292,0.345651]
        criterion = [0.325804,0.325804,0.325804,0.325804,0.325804,0.325804,0.325804]
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
    x_axix = range(len(LFI))

    plt.plot(x_axix, LFI, color=mycolor[0], label='NANG', marker=mymarker[0], linewidth=my_line_width,
             markersize=my_marker_size)
    plt.plot(x_axix, criterion, color=mycolor[4], label='GCNs', linestyle='--', linewidth=my_line_width,
             markersize=my_marker_size)

    my_legend = plt.legend(loc='best', fontsize=12)
    frame = my_legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')

    plt.xticks(x_axix, [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
    plt.tick_params(labelsize='10')
    plt.xlabel('$\lambda_{c}$', fontdict=font)
    if dataset=='steam':
        plt.ylabel('Recall@5', fontdict=font)
    else:
        plt.ylabel('Recall@20', fontdict=font)
    # plt.show()
    plt.savefig(os.path.join(os.getcwd(), 'figures', 'LFI', '{}_lambda_c_recall.png'.format(dataset)))

elif plot_choice == 'N':
    # plot for NDCG
    if dataset=='cora':
        LFI = [0.228883,0.236803,0.24632,0.251469,0.254618,0.259858,0.255438]
        criterion = [0.207984,0.207984,0.207984,0.207984,0.207984,0.207984,0.207984]
    elif dataset=='citeseer':
        LFI = [0.136691,0.16377,0.164586,0.167747,0.172966,0.178501,0.184793]
        criterion = [0.142371,0.142371,0.142371,0.142371,0.142371,0.142371,0.142371]
    elif dataset=='steam':
        LFI = [0.409084,0.42055,0.422709,0.426792,0.433209,0.429652,0.425394]
        criterion = [0.402568,0.402568,0.402568,0.402568,0.402568,0.402568,0.402568]
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
    x_axix = range(len(LFI))

    plt.plot(x_axix, LFI, color=mycolor[0], label='NANG', marker=mymarker[0], linewidth=my_line_width,
             markersize=my_marker_size)
    plt.plot(x_axix, criterion, color=mycolor[4], label='GCNs', linestyle='--', linewidth=my_line_width,
             markersize=my_marker_size)

    my_legend = plt.legend(loc='best', fontsize=12)
    frame = my_legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')

    plt.xticks(x_axix, [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
    plt.tick_params(labelsize='10')
    plt.xlabel('$\lambda_{c}$', fontdict=font)
    if dataset=='steam':
        plt.ylabel('NDCG@5', fontdict=font)
    else:
        plt.ylabel('NDCG@20', fontdict=font)
    # plt.show()
    plt.savefig(os.path.join(os.getcwd(), 'figures', 'LFI', '{}_lambda_c_ndcg.png'.format(dataset)))
