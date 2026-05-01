import pandas as pd
import numpy as np
from mp_api.client import MPRester
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition



class OxyData():
    def __init__(self, api):
        self.api = api
        self.df = None

    def process_data(self, max_elements=6):
        with MPRester(self.api) as mpr:
            data_init = []

            for n in range(2, max_elements + 1):
                data = mpr.materials.summary.search(elements=['O'], num_elements=n,
                                                             fields=['formula_pretty', 'formation_energy_per_atom',
                                                                     'energy_above_hull'])
                data_init.extend(data)

        # CREATING DATAFRAME FOR DATA

        data = data_init
        tmp_dict = []
        for d in data:
            tmp_dict.append({
                'formula_pretty': d.formula_pretty,
                'formation_energy_per_atom': d.formation_energy_per_atom,
                'energy_above_hull': d.energy_above_hull
            })

        magpie = ElementProperty.from_preset('magpie')
        conv = StrToComposition()

        data = pd.DataFrame(tmp_dict,
                            columns=['formula_pretty', 'formation_energy_per_atom', 'structure', 'energy_above_hull'])

        # APPEND COMPOSITION TO DATAFRAME

        df = conv.featurize_dataframe(data, 'formula_pretty')

        # MAGPIE

        df = magpie.featurize_dataframe(df, 'composition')
        self.df = df

    def save_to_file(self):
        if self.df is not None:
            self.df.to_pickle('processed_data.pkl')
            print('Data saved.')

    def load_data(self):
        self.df = pd.read_pickle('processed_data.pkl')