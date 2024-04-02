from qsprpred.data.chem.scaffolds import BemisMurcko
from qsprpred.data.descriptors.fingerprints import MorganFP
from scaffviz.clustering.manifold import TSNE
from qsprpred.data.sources.papyrus import Papyrus
from scaffviz.depiction.plot import Plot

acc_keys = ["P51681"]  # accession keys of the proteins to fetch data for
name = "P51681_LIGANDS_nostereo"  # name of the data set
quality = "low"  # choose minimum quality from {"high", "medium", "low"}

# fetch data from Papyrus for the given targets (may take a while)
papyrus = Papyrus(
    data_dir="./data",  # where to store the data
    stereo=False,  # do not consider stereochemistry
    plus_only=True,  # only use the high quality data set
    descriptors=None,  # do not download descriptors (we will calculate them later)
)
dataset = papyrus.getData(
    name,
    acc_keys,
    quality,
    use_existing=True # use existing data set if it was already compiled before
)

# add generic scaffolds to the data set
# these will be used to group the molecules
dataset.addScaffolds([BemisMurcko()])

# add Morgan fingerprints to the data set
# these will be used to calculate the t-SNE embedding in 2D
dataset.addDescriptors([MorganFP(radius=3, nBits=2048)], recalculate=False)

# make an interactive plot that will use t-SNE to embed the data set in 2D
# (all available descriptors in the data set will be used, not just selected features)
plt = Plot(TSNE(perplexity=150))
plt.plot(
    dataset,
    recalculate=False,  # do not recalculate the t-SNE embedding each time this is run
    mols_per_scaffold_group=5,  # smaller groups will be merged into 'Other' group
    card_data=["all_doc_ids"],  # what to show on the molecule cards
    title_data='InChIKey'   # Data to show in the title of the molecule cards
)