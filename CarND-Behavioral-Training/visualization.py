import matplotlib.pyplot as plt


def hist(dataset, attr, title=None, xlabel=None, ylabel=None, grid=True):
    plt.hist(dataset[attr].values, 100, normed=0, alpha=0.75)
    plt.title(title)
    if grid is not True:
        plt.ylabel(xlabel)
        plt.xlabel(ylabel)
    plt.show()


def plots(images, row=None, col=None, figsize=(8, 6), labels=[], grid=True):
    if (row is None or col is None):
        row = len(images)
        col = 1

    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(row, col, i+1)
        plt.imshow(img)

        if grid is False:
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

        if len(labels) == len(images):
            plt.xlabel(labels[i])

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    pass