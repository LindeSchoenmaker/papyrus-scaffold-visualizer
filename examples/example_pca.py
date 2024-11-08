"""
example_coloring

Created by: Martin Sicho
On: 02.11.22, 14:09
"""
from qsprpred.data.descriptors.sets import DrugExPhyschem
from utils import fetch_example_dataset

from scaffviz.clustering.manifold import PCA
from scaffviz.depiction.plot import Plot

if __name__ == '__main__':
    # load data
    dataset = fetch_example_dataset()
    dataset.addDescriptors([DrugExPhyschem()], recalculate=False)
    # plot
    plt = Plot(PCA())
    plt.plot(
        dataset,
        recalculate=False,
        mols_per_scaffold_group=5,
        port=9292
    )
