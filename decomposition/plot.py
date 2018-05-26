import matplotlib.pyplot as plt

print('Starting')
def plot_stuff(results, dataset, x_axis):
    for decomposer in results:
        precision = results[decomposer]['precision']
        recall = results[decomposer]['recall']
        f1 = results[decomposer]['f1']
        
        plt.figure(1)

        # Precision
        plt.subplot(311)
        plt.plot(x_axis, precision, 'k', x_axis, precision, 'r^')
        plt.yscale('linear', linthreshy=1)
        plt.title(decomposer + ' on ' + dataset)
        plt.xlabel('Dimension')
        plt.ylabel('Precision mean')
        plt.grid(True)

        # Recall
        plt.subplot(312)
        plt.plot(x_axis, recall, 'k', x_axis, recall, 'ro')
        plt.yscale('linear', linthreshy=1)
        plt.xlabel('Dimension')
        plt.ylabel('Recall mean')
        plt.grid(True)

        # F1
        plt.subplot(313)
        plt.plot(x_axis, f1, 'k', x_axis, f1, 'rs')
        plt.yscale('linear', linthreshy=1)
        plt.xlabel('Dimension')
        plt.ylabel('F1 mean')
        plt.grid(True)

        # plt.axis([1, 5, 0, 1])
        print('Ploting')
        plt.show()