import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(lines, shapes, colors, labels, markers,
                        save_path, title='', logy=False,
                        ms=5., linewidth=5.,
                        xlabel=None, ylabel=None,
                        ylim=None, yticks=None, xlim=None, xticks=None, xtick_label=None, ytick_label=None,
                        legend_font=12., legend_loc='upper right'):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel, fontsize=24); plt.ylabel(ylabel, fontsize=24)
    if xlim or xticks: plt.xticks(xticks, xtick_label, fontsize=24); plt.xlim(xlim)
    if ylim or yticks: plt.yticks(yticks, ytick_label, fontsize=24); plt.ylim(ylim);
    plt.grid(linestyle='dotted',axis='y')

    for idx, line in enumerate(lines):
        if not logy:
            plt.plot(np.arange(len(line)), line, shapes[idx], color=colors[idx], label=labels[idx],
                 linewidth=linewidth, marker=markers[idx], ms=ms)
        else:
            plt.semilogy(np.arange(len(line)), line, shapes[idx], color=colors[idx], label=labels[idx],
                 linewidth=linewidth, marker=markers[idx], ms=ms)

    plt.legend(loc=legend_loc, fontsize=legend_font)
    plt.savefig(save_path, dpi=900, bbox_inches='tight', pad_inches=0)

def plot_learning_curve_uneven(lines, shapes, colors, labels, markers,
                        save_path, title='', logy=False,
                        ms=5., linewidth=5.,
                        xlabel=None, ylabel=None,
                        ylim=None, yticks=None, xlim=None, xticks=None, xtick_label=None, ytick_label=None,
                        legend_font=12., legend_loc='upper right'):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel, fontsize=24); plt.ylabel(ylabel, fontsize=24)
    if xlim or xticks: plt.xticks(xticks, xtick_label, fontsize=24); plt.xlim(xlim)
    if ylim or yticks: plt.yticks(yticks, ytick_label, fontsize=24); plt.ylim(ylim);
    plt.grid(linestyle='dotted',axis='y')

    for idx, line in enumerate(lines):
        if not logy:
            plt.plot(xticks, line, shapes[idx], color=colors[idx], label=labels[idx],
                     linewidth=linewidth, marker=markers[idx], ms=ms)
        else:
            plt.semilogy(xticks, line, shapes[idx], color=colors[idx], label=labels[idx],
                         linewidth=linewidth, marker=markers[idx], ms=ms)

    plt.legend(loc=legend_loc, fontsize=legend_font)
    plt.savefig(save_path, dpi=900, bbox_inches='tight', pad_inches=0)

# vary m
if True:
    basic_solution = np.array([822689.788, 1330343.275, 2019677.863, 3768041.58, 6168518.202, 9092962.537])/(60000)
    basic_solution_pp = np.array([672001.545, 981328.654, 1405888.996, 2310591.49, 3598752.027, 4914150.694])/(60000)
    enhanced_solution = np.array([8759830.066, 11632274.13, 14547773.78, 20880090.19, 28016917.77, 35933279.48])/(60000)
    enhanced_solution_pp = np.array([4976876.39, 5775811.063, 6812708.759, 8636597.903, 11084675.21, 13384859.64])/(60000)

    lines = [basic_solution, basic_solution_pp, enhanced_solution, enhanced_solution_pp]
    shapes = ['-', '-.', '-', '-.']
    markers = ['s', 'o', 'D', '^']
    labels = ['Pivot-Basic', 'Pivot-Basic-PP', 'Pivot-Enhanced', 'Pivot-Enhanced-PP']
    #colors = ['#009933', '#66ff33', '#0033cc', '#0066ff']
    colors = ['b', 'g', 'r', 'm']
    xlim = [1.8, 10.2]
    ylim = [0, 800]
    #xticks = range(5)
    xticks = [2, 3, 4, 6, 8, 10]
    yticks = [0, 200, 400, 600, 800]
    xtick_label = ['2', '3', '4', '6', '8', '10']
    ytick_label = ['0', '200', '400', '600', '800']

    plot_learning_curve_uneven(lines=lines, shapes=shapes, colors=colors, labels=labels, markers=markers,
                        save_path='figs/vary_m.pdf',
                        title='', logy=False, ms=12, linewidth=3,
                        xlabel='$m$', ylabel='Training Time (min)', ylim=ylim, yticks=yticks,
                        xlim=xlim, xticks=xticks, xtick_label=xtick_label, ytick_label=ytick_label, legend_font=16, legend_loc='upper left')

# vary n
if True:
    basic_solution = np.array([1160786.092, 1176500.369, 1330343.275, 1587529.102, 2076230.819])/(60000)
    basic_solution_pp = np.array([764980.12, 782337.539, 981328.654, 1222573.492, 1711838.596])/(60000)
    enhanced_solution = np.array([2182740.249, 3214100.898, 11632274.13, 22114537.05, 42753624.01])/(60000)
    enhanced_solution_pp = np.array([1264843.094, 1771361.828, 5775811.063, 10815403.56, 20837979.56])/(60000)

    lines = [basic_solution, basic_solution_pp, enhanced_solution, enhanced_solution_pp]
    shapes = ['-', '-.', '-', '-.']
    markers = ['s', 'o', 'D', '^']
    labels = ['Pivot-Basic', 'Pivot-Basic-PP', 'Pivot-Enhanced', 'Pivot-Enhanced-PP']
    #colors = ['#009933', '#66ff33', '#0033cc', '#0066ff']
    colors = ['b', 'g', 'r', 'm']
    xlim = [-0.2, 4.2]
    ylim = [0, 800]
    xticks = range(5)
    yticks = [0, 200, 400, 600, 800]
    xtick_label = ['5k', '10k', '50k', '100k', '200k']
    ytick_label = ['0', '200', '400', '600', '800']

    plot_learning_curve(lines=lines, shapes=shapes, colors=colors, labels=labels, markers=markers,
                        save_path='figs/vary_n.pdf',
                        title='', logy=False, ms=12, linewidth=3,
                        xlabel='$n$', ylabel='Training Time (min)', ylim=ylim, yticks=yticks,
                        xlim=xlim, xticks=xticks, xtick_label=xtick_label, ytick_label=ytick_label, legend_font=16, legend_loc='upper left')

# vary d
if True:
    basic_solution = np.array([508702.922, 1330343.275, 2523424.864, 4929702.87, 9991561.719])/(60000)
    basic_solution_pp = np.array([409336.699, 981328.654, 1880478.914, 3583171.214, 6827141.886])/(60000)
    enhanced_solution = np.array([10747758.66, 11632274.13, 12871815.88, 15363180.18, 20376077.21])/(60000)
    enhanced_solution_pp = np.array([5208559.656, 5775811.063, 6667274.732, 8457606.715, 11900533.43])/(60000)

    lines = [basic_solution, basic_solution_pp, enhanced_solution, enhanced_solution_pp]
    shapes = ['-', '-.', '-', '-.']
    markers = ['s', 'o', 'D', '^']
    labels = ['Pivot-Basic', 'Pivot-Basic-PP', 'Pivot-Enhanced', 'Pivot-Enhanced-PP']
    colors = ['b', 'g', 'r', 'm']
    xlim = [-0.2, 4.2]
    ylim = [0, 400]
    xticks = range(5)
    yticks = [0, 100, 200, 300, 400]
    xtick_label = ['5', '15', '30', '60', '120']
    ytick_label = ['0', '100', '200', '300', '400']

    plot_learning_curve(lines=lines, shapes=shapes, colors=colors, labels=labels, markers=markers,
                        save_path='figs/vary_d.pdf',
                        title='', logy=False, ms=12, linewidth=3,
                        xlabel='$\\bar{d}$', ylabel='Training Time (min)', ylim=ylim, yticks=yticks,
                        xlim=xlim, xticks=xticks, xtick_label=xtick_label, ytick_label=ytick_label, legend_font=16, legend_loc='upper left')

# vary split_num
if True:
    basic_solution = np.array([374593.067, 690494.683, 1330343.275, 2586320.03, 5018416.149])/(60000)
    basic_solution_pp = np.array([335274.584, 558495.617, 981328.654, 1826438.97, 3560476.637])/(60000)
    enhanced_solution = np.array([10567803.76, 10901940.05, 11632274.13, 12768027.69, 15257682.35])/(60000)
    enhanced_solution_pp = np.array([5135410.606, 5351739.312, 5775811.063, 6668996.846, 8397098.167])/(60000)

    lines = [basic_solution, basic_solution_pp, enhanced_solution, enhanced_solution_pp]
    shapes = ['-', '-.', '-', '-.']
    markers = ['s', 'o', 'D', '^']
    labels = ['Pivot-Basic', 'Pivot-Basic-PP', 'Pivot-Enhanced', 'Pivot-Enhanced-PP']
    colors = ['b', 'g', 'r', 'm']
    xlim = [-0.2, 4.2]
    ylim = [0, 400]
    xticks = range(5)
    yticks = [0, 100, 200, 300, 400]
    xtick_label = ['2', '4', '8', '16', '32']
    ytick_label = ['0', '100', '200', '300', '400']

    plot_learning_curve(lines=lines, shapes=shapes, colors=colors, labels=labels, markers=markers,
                        save_path='figs/vary_splitNum.pdf',
                        title='', logy=False, ms=12, linewidth=3,
                        xlabel='$b$', ylabel='Training Time (min)', ylim=ylim, yticks=yticks,
                        xlim=xlim, xticks=xticks, xtick_label=xtick_label, ytick_label=ytick_label, legend_font=16, legend_loc='upper left')


# vary class num
if True:
    basic_solution = np.array([842635.837, 1330343.275, 2286192.96, 4225067.674, 5980867.946])/(60000)
    basic_solution_pp = np.array([638548.696, 981328.654, 1660671.765, 2956065.642, 4098042.953])/(60000)
    enhanced_solution = np.array([11047351.27, 11632274.13, 12716668.54, 14390669.31, 17188417.09])/(60000)
    enhanced_solution_pp = np.array([5448479.878, 5775811.063, 6472209.304, 7821108.251, 10437695.44])/(60000)

    lines = [basic_solution, basic_solution_pp, enhanced_solution, enhanced_solution_pp]
    shapes = ['-', '-.', '-', '-.']
    markers = ['s', 'o', 'D', '^']
    labels = ['Pivot-Basic', 'Pivot-Basic-PP', 'Pivot-Enhanced', 'Pivot-Enhanced-PP']
    colors = ['b', 'g', 'r', 'm']
    xlim = [-0.2, 4.2]
    ylim = [0, 400]
    xticks = range(5)
    yticks = [0, 100, 200, 300, 400]
    xtick_label = ['2', '4', '8', '16', '32']
    ytick_label = ['0', '100', '200', '300', '400']

    plot_learning_curve(lines=lines, shapes=shapes, colors=colors, labels=labels, markers=markers,
                        save_path='figs/vary_classNum.pdf',
                        title='', logy=False, ms=12, linewidth=3,
                        xlabel='$c$', ylabel='Training Time (min)', ylim=ylim, yticks=yticks,
                        xlim=xlim, xticks=xticks, xtick_label=xtick_label, ytick_label=ytick_label, legend_font=16, legend_loc='upper left')


# vary tree_depth
if False:
    basic_solution = np.array([352731.551, 694191.652, 1330343.275, 2634886.799, 5134124.694])/(60000)
    basic_solution_pp = np.array([277186.347, 538449.966, 981328.654, 1973280.671, 3832426.848])/(60000)
    enhanced_solution = np.array([2389211.304, 5487717.84, 11632274.13, 23730182.92, 48868967.21])/(60000)
    enhanced_solution_pp = np.array([1246909.41, 2776216.228, 5775811.063, 11928529.71, 24082958.42])/(60000)

    lines = [basic_solution, basic_solution_pp, enhanced_solution, enhanced_solution_pp]
    shapes = ['-', '-.', '-', '-.']
    markers = ['s', 'o', 'D', '^']
    labels = ['Pivot-Basic', 'Pivot-Basic-PP', 'Pivot-Enhanced', 'Pivot-Enhanced-PP']
    colors = ['b', 'g', 'r', 'm']
    xlim = [-0.2, 4.2]
    ylim = [0, 850]
    xticks = range(5)
    yticks = [0, 200, 400, 600, 800]
    xtick_label = ['2', '3', '4', '5', '6']
    ytick_label = ['0', '200', '400', '600', '800']

    plot_learning_curve(lines=lines, shapes=shapes, colors=colors, labels=labels, markers=markers,
                        save_path='figs/vary_treeDepth.pdf',
                        title='', logy=False, ms=12, linewidth=3,
                        xlabel='$h$', ylabel='Training Time (min)', ylim=ylim, yticks=yticks,
                        xlim=xlim, xticks=xticks, xtick_label=xtick_label, ytick_label=ytick_label, legend_font=16, legend_loc='upper left')

# vary numTree
if True:
    basic_solution_RF_PP_Classification = np.array([2470255.588, 5032626.445, 9892574.406, 19983417.36, 39799246.32]) / (60000*60)
    #basic_solution_GBDT_PP_Classification = np.array([70764000.23, 141634146.5, 282783904.1, 566458577.6, 1132917155]) / (60000*60)
    basic_solution_GBDT_PP_Classification = np.array([11882243.96, 28924962.71, 60441964.24, 124446185.3, 253952848.4]) / (60000*60)
    basic_solution_RF_PP_Regression = np.array([1214161.479, 2450498.055, 4945367.987, 9879509.475, 19538515]) / (60000*60)
    #basic_solution_GBDT_PP_Regression = np.array([17237523.9, 35336923.99, 70673847.98, 141347696, 282695391.9]) / (60000*60)
    basic_solution_GBDT_PP_Regression = np.array([1882710.232, 3654598.904, 7235154.055, 14451409.94, 28908112.76]) / (60000*60)

    lines = [basic_solution_RF_PP_Classification, basic_solution_GBDT_PP_Classification, basic_solution_RF_PP_Regression, basic_solution_GBDT_PP_Regression]
    #lines = [basic_solution_RF_PP_Classification, basic_solution_RF_PP_Regression, basic_solution_GBDT_PP_Regression]
    shapes = ['-', '-', '-', '-']
    markers = ['s', 'o', 'D', '^']
    labels = ['Pivot-RF-Classification', 'Pivot-GBDT-Classification', 'Pivot-RF-Regression', 'Pivot-GBDT-Regression']
    colors = ['b', 'g', 'r', 'm']
    xlim = [-0.2, 4.2]
    ylim = [0, 80]
    xticks = range(5)
    yticks = [0, 20, 40, 60, 80]
    xtick_label = ['2', '4', '8', '16', '32']
    ytick_label = ['0', '20', '40', '60', '80']

    plot_learning_curve(lines=lines, shapes=shapes, colors=colors, labels=labels, markers=markers,
                        save_path='figs/vary_numTree.pdf',
                        title='', logy=False, ms=12, linewidth=3,
                        xlabel='$W$', ylabel='Training Time (hour)', ylim=ylim, yticks=yticks,
                        xlim=xlim, xticks=xticks, xtick_label=xtick_label, ytick_label=ytick_label, legend_font=16, legend_loc='upper left')


# vary numTree prediction
if True:
    basic_solution_RF_PP_Classification = np.array([29.743,59.385,118.347,237.233,475.607])
    basic_solution_GBDT_PP_Classification = np.array([119.346,239.892,479.242,958.484,1916.97])
    basic_solution_RF_PP_Regression = np.array([29.583,59.97,118.923,237.092,472.197])
    basic_solution_GBDT_PP_Regression = np.array([29.739,59.347,117.963,237.438,472.126])

    lines = [basic_solution_RF_PP_Classification, basic_solution_GBDT_PP_Classification, basic_solution_RF_PP_Regression, basic_solution_GBDT_PP_Regression]
    #lines = [basic_solution_RF_PP_Classification, basic_solution_RF_PP_Regression, basic_solution_GBDT_PP_Regression]
    shapes = ['-', '-', '-', '-']
    markers = ['s', 'o', 'D', '^']
    labels = ['Pivot-RF-Classification', 'Pivot-GBDT-Classification', 'Pivot-RF-Regression', 'Pivot-GBDT-Regression']
    colors = ['b', 'g', 'r', 'm']
    xlim = [-0.2, 4.2]
    ylim = [0, 2000]
    xticks = range(5)
    yticks = [0, 500, 1000, 1500, 2000]
    xtick_label = ['2', '4', '8', '16', '32']
    ytick_label = ['0', '500', '1000', '1500', '2000']

    plot_learning_curve(lines=lines, shapes=shapes, colors=colors, labels=labels, markers=markers,
                        save_path='figs/vary_numTree_prediction.pdf',
                        title='', logy=False, ms=12, linewidth=3,
                        xlabel='$W$', ylabel='Prediction Time (ms)', ylim=ylim, yticks=yticks,
                        xlim=xlim, xticks=xticks, xtick_label=xtick_label, ytick_label=ytick_label, legend_font=16, legend_loc='upper left')


# vary m_2
if False:
    basic_solution = np.array([9.884, 14.132, 19.003, 27.878, 37.125, 46.541])
    enhanced_solution = np.array([37.624, 38.978, 39.862, 41.612, 45.503, 47.937])
    npddt_solution = np.array([0.191, 0.217, 0.27, 0.314, 0.381, 0.371])

    lines = [basic_solution, enhanced_solution, npddt_solution]
    shapes = ['-', '-', '-']
    markers = ['s', 'D', 'v']
    labels = ['Pivot-Basic', 'Pivot-Enhanced', 'NPD-DT']
    colors = ['b', 'r', 'm']
    xlim = [1.8, 10.2]
    ylim = [0, 60]
    #xticks = range(5)
    xticks = [2, 3, 4, 6, 8, 10]
    yticks = [0, 20, 40, 60]
    xtick_label = ['2', '3', '4', '6', '8', '10']
    ytick_label = ['0', '20', '40', '60']

    plot_learning_curve_uneven(lines=lines, shapes=shapes, colors=colors, labels=labels, markers=markers,
                        save_path='figs/vary_m_prediction.pdf',
                        title='', logy=False, ms=12, linewidth=3,
                        xlabel='$m$', ylabel='Prediction Time (ms)', ylim=ylim, yticks=yticks,
                        xlim=xlim, xticks=xticks, xtick_label=xtick_label, ytick_label=ytick_label, legend_font=16, legend_loc='upper left')

# vary tree_depth_2
if False:
    basic_solution = np.array([13.896, 13.963, 14.132, 14.437, 15.211])
    enhanced_solution = np.array([8.022, 17.908, 38.978, 76.721, 158.525])
    npddt_solution = np.array([0.006, 0.091, 0.217, 0.245, 0.354])

    lines = [basic_solution, enhanced_solution, npddt_solution]
    shapes = ['-', '-', '-']
    markers = ['s', 'D', 'v']
    labels = ['Pivot-Basic', 'Pivot-Enhanced', 'NPD-DT']
    colors = ['b', 'r', 'm']
    xlim = [-0.2, 4.2]
    ylim = [0, 165]
    xticks = range(5)
    yticks = [0, 40, 80, 120, 160]
    xtick_label = ['2', '3', '4', '5', '6']
    ytick_label = ['0', '40', '80', '120', '160']

    plot_learning_curve(lines=lines, shapes=shapes, colors=colors, labels=labels, markers=markers,
                        save_path='figs/vary_treeDepth_prediction.pdf',
                        title='', logy=False, ms=12, linewidth=3,
                        xlabel='$h$', ylabel='Prediction Time (ms)', ylim=ylim, yticks=yticks,
                        xlim=xlim, xticks=xticks, xtick_label=xtick_label, ytick_label=ytick_label, legend_font=16, legend_loc='upper left')

# vary num_tree_2
if False:
    basic_solution_RF_PP = np.array([29.177, 57.602, 116.529, 237.042, 461.656])
    basic_solution_GBDT_PP = np.array([116.208, 232.947, 461.657, 923.313, 1846.628])

    lines = [basic_solution_RF_PP, basic_solution_GBDT_PP]
    shapes = ['-', '-.']
    markers = ['v', '^']
    labels = ['Pivot-Basic-RF-PP', 'Pivot-Basic-GBDT-PP']
    colors = ['#0033cc', '#0066ff']
    xlim = [-0.2, 4.2]
    ylim = None
    xticks = range(5)
    xtick_label = ['2', '4', '8', '16', '32']

    plot_learning_curve(lines=lines, shapes=shapes, colors=colors, labels=labels, markers=markers,
                        save_path='figs/vary_numTree_2.pdf',
                        title='', logy=False, ms=3, linewidth=0.8,
                        xlabel='Tree Number', ylabel='Training Time (ms)', ylim=ylim, yticks=None,
                        xlim=xlim, xticks=xticks, xtick_label=xtick_label, legend_font=8, legend_loc='upper left')

# vary m_3
if False:
    basic_solution = np.array([822689.788, 1330343.275, 2019677.863, 3768041.58, 6168518.202, 9092962.537])/(60000)
    enhanced_solution = np.array([8759830.066, 11632274.13, 14547773.78, 20880090.19, 28016917.77, 35933279.48])/(60000)
    SPDZ_DZ = np.array([8307152.885, 19876233.69, 35191066.55, 74834308.21, 115319597.9, 162280010.6])/(60000)
    Non_private = np.array([1462.867, 1982.675, 2514.447, 3545.563, 4606.161, 5642.511])/(60000)

    lines = [basic_solution, enhanced_solution, SPDZ_DZ, Non_private]
    shapes = ['-', '-', '-', '-']
    markers = ['s', 'D', 'P', 'v']
    labels = ['Pivot-Basic', 'Pivot-Enhanced', 'SPDZ-DT', 'NPD-DT']
    colors = ['b', 'r', 'c', 'm']
    xlim = [1.8, 10.2]
    ylim = [0, 3000]
    #xticks = range(6)
    xticks = [2, 3, 4, 6, 8, 10]
    yticks = [0, 1000, 2000, 3000]
    xtick_label = ['2', '3', '4', '6','8', '10']
    ytick_label = ['0', '1000', '2000', '3000']

    plot_learning_curve_uneven(lines=lines, shapes=shapes, colors=colors, labels=labels, markers=markers,
                        save_path='figs/vary_m_comparison.pdf',
                        title='', logy=False, ms=12, linewidth=3,
                        xlabel='$m$', ylabel='Training Time (min)', ylim=ylim, yticks=yticks,
                        xlim=xlim, xticks=xticks, xtick_label=xtick_label, ytick_label=ytick_label, legend_font=16, legend_loc='upper left')

# vary n_2
if False:
    basic_solution = np.array([1160786.092, 1176500.369, 1330343.275, 1587529.102, 2076230.819])/(60000)
    enhanced_solution = np.array([2182740.249, 3214100.898, 11632274.13, 22114537.05, 42753624.01])/(60000)
    #SPDZ_DZ = np.array([1950356.105, 5102164.833, 19876233.69, 57118889.29, 77949970.76])/(60000)
    SPDZ_DZ = np.array([1950356.105, 5102164.833, 21769667.48, 57118889.29, 81954667.13])/(60000)
    non_private = np.array([657.691, 796.659, 1982.675, 3482.41, 6562.941])/(60000)

    lines = [basic_solution, enhanced_solution, SPDZ_DZ, non_private]
    shapes = ['-', '-', '-', '-']
    markers = ['s', 'D', 'P', 'v']
    labels = ['Pivot-Basic', 'Pivot-Enhanced', 'SPDZ-DT', 'NPD-DT']
    colors = ['b', 'r', 'c', 'm']
    xlim = [-0.2, 4.2]
    ylim = [0, 1600]
    xticks = range(5)
    yticks = [0, 400, 800, 1200, 1600]
    xtick_label = ['5k', '10k', '50k', '100k', '200k']
    ytick_label = ['0', '400', '800', '1200', '1600']

    plot_learning_curve(lines=lines, shapes=shapes, colors=colors, labels=labels, markers=markers,
                        save_path='figs/vary_n_comparison.pdf',
                        title='', logy=False, ms=12, linewidth=3,
                        xlabel='$n$', ylabel='Training Time (min)', ylim=ylim, yticks=yticks,
                        xlim=xlim, xticks=xticks, xtick_label=xtick_label, ytick_label=ytick_label, legend_font=16, legend_loc='upper left')




