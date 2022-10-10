# Papyrus Scaffold Visualizer

Extraction and interactive visualization of data extracted from [Papyrus](https://chemrxiv.org/engage/chemrxiv/article-details/617aa2467a002162403d71f0). This package allows you to quickly filter out relevant data from the Papyrus database and make an interactive visualization of the compounds in the data set. By default, the compounds are grouped and colored by scaffold, but you can also customize the grouping and coloring by other means. If you want to quickly analyze what kind of compounds you have in the data set, this can help you to do so effortlessly. The assembled data set is also a good starting point for machine learning.


## Getting started

You can install the package from PyPI:

```bash
   pip install papyrus-scaffold-visualizer
```

You can then use the package to extract data from the Papyrus database and visualize it ([examples/example_depiction_papyrus.py](./examples/example_depiction_papyrus.py)):

```python
from scaffviz.clustering.descriptors import MorganFP
from scaffviz.clustering.manifold import TSNE
from scaffviz.clustering.scaffolds import Murcko
from scaffviz.data.papyrus.papyrus_class import Papyrus
from scaffviz.depiction.plot import Plot

if __name__ == "__main__":

    acc_keys = ["P41597"] # replace with your own accession key
    name = "P41597_LIGANDS_nostereo" # replace with your own name for the output data set file
    quality = "low" # choose minimum quality from {"high", "medium", "low"}

    # fetches latest version of Papyrus if not already available and filters out the relevant data
    papyrus = Papyrus(data_dir="./data", stereo=False)
    dataset = papyrus.getData(
        acc_keys, 
        quality, 
        name, 
        use_existing=True # use existing data set if it was already compiled before (set to False to fetch data again)
    )
    
    # use the filtered data to create the visualization
    dataset.addScaffolds([Murcko()]) # add Murcko scaffolds to the data set -> will be used to group compounds
    dataset.addDescriptors([MorganFP(radius=2, nBits=1024)], recalculate=False) # add Morgan fingerprints to the data set -> will be used to calculate the t-SNE embedding

    plt = Plot(dataset, TSNE()) # make an interactive plot that will use t-SNE to embed the data set in 2D (all descriptors in the data set will be used)
    plt.plot(recalculate=False, mols_per_scaffold_group=5) # start the server, you can open the plot in the browser
```

You can find more example scripts under [examples](./examples).

## License
[MIT License](./LICENSE.md).

## Credits

This work is based on the code developed by Nadieh van Marrewijk and Yorick van Aalst during their internships at the Computational Drug Discovery Group, LACDR, Leiden University.
