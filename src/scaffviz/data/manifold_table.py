"""
manifold_table

Created by: Martin Sicho
On: 17.01.23, 17:20
"""
import numpy as np
import pandas as pd
from qsprpred.data import MoleculeTable
from qsprpred.data.storage.tabular.basic_storage import PandasChemStore

from scaffviz.clustering.manifold import Manifold


class ManifoldTable(MoleculeTable):

    @classmethod
    def fromDF(
        cls,
        name: str,
        df: pd.DataFrame,
        path: str = ".",
        smiles_col: str = "SMILES",
        **kwargs,
    ) -> "MoleculeTable":
        """Create a `MoleculeTable` instance from a pandas DataFrame.

        Args:
            name (str): Name of the data set.
            df (pd.DataFrame): DataFrame containing the molecule data.
            path (str): Path to the directory where the data set will be stored.
            smiles_col (str): Name of the column in the data frame containing the SMILES
                sequences.
            **kwargs:
                Additional keyword arguments to pass to the `MoleculeTable` constructor.

        Returns:
            (MoleculeTable): The created data set.
        """
        storage = PandasChemStore(
            f"{name}_storage", path, df, smiles_col=smiles_col, **kwargs
        )
        return ManifoldTable(storage, name=name, path=path)


    @staticmethod
    def fromMolTable(mol_table : MoleculeTable, name=None):
        name = name if name is not None else mol_table.name
        mt = ManifoldTable.fromDF(name, mol_table.getDF(), smiles_col=mol_table.smilesProp, path=mol_table.path)
        mt.descriptors = mol_table.descriptors
        return mt

    def getManifoldData(self, manifold: Manifold):
        if str(manifold) in self.getSubset(str(manifold)):
            return self.getSubset(str(manifold))
        else:
            return None

    def addManifoldData(self, manifold : Manifold, recalculate=True):
        manifold_data = self.getManifoldData(manifold)
        manifold_cols = []
        if manifold_data is not None:
            manifold_cols = manifold_data.columns.tolist()
        if recalculate or manifold_data is None:
            if not self.hasDescriptors():
                raise ValueError("Descriptors must be calculated before adding manifold data.")
            X = manifold.fit_transform(self.getDescriptors())
            manifold_cols = []
            x = np.transpose(X)
            for i, dim in enumerate(x):
                col_name = f"{manifold}_{i + 1}"
                manifold_cols.append(col_name)
                self.addProperty(col_name, dim)

        return manifold_cols