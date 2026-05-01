import pandas as pd
import numpy as np
from mp_api.client import MPRester
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition
import itertools

# FUNCTIONS

def OxyScreener(support_form, model_class, model_reg):
    supp_dict = {'formula_pretty': [support_form]}

    # Create Dataframe
    supp_df = pd.DataFrame(supp_dict, columns=['formula_pretty'])

    # Magpie
    supp_df = conv.featurize_dataframe(supp_df, 'formula_pretty', pbar=False)
    supp_df = magpie.featurize_dataframe(supp_df, 'composition', pbar=False)

    # Prepare data
    X_supp = supp_df.drop(columns=['formula_pretty', 'composition'])

    # OUTPUT

    proba = model_class.predict_proba(X_supp)
    stability = model_class.predict(X_supp)

    print(f"")
    print(f'=== {support_form} ===')
    print(f'Stability: {stability[0]}')

    energy_supp = 0

    if proba[0][0] > proba[0][1]:
        print(f'Confidence: {100 * proba[0][0]:.2f}%')  # SWITCHES ON FOR UNSTABLE SUPPORTS
    else:
        print(f'Confidence: {100 * proba[0][1]:.2f}%')  # SWITCHES ON FOR STABLE SUPPORTS

    if stability[0] == 1:
        energy = model_reg.predict(X_supp)
        energy_supp = energy[0]
        print(f'Energy: {energy[0]:.2f}')

    return {
        'formula': support_form,
        'stability': stability[0],
        'confidence_stable': proba[0][1],
        'energy': energy_supp
    }


def generate_list(fixed_dict, scan_list, o_number=2, step = 0.1):
    """
    fixed_dict = {'Ce': 0.4, 'Zr': 0.3}
    scan_list = ['La', 'Eu', 'Gd']
    """
    total_fixed = sum(fixed_dict.values())
    remaining_budget = round(1.0 - total_fixed, 2)

    if remaining_budget < 0:
        print("ERROR: Fixed concentrations exceed 1.0!")
        sys.exit()

    grid = np.round(np.arange(0, remaining_budget + 0.0001, step), 2)

    formulas = []
    for weights in itertools.product(grid, repeat=len(scan_list)):
        if np.isclose(sum(weights), remaining_budget):

            composition = fixed_dict.copy()
            for i, element in enumerate(scan_list):
                composition[element] = weights[i]

            formula_str = ""
            for el, val in composition.items():
                if val > 0:
                    formula_str += f"{el}{val:.2f}"

            formula_str += f"O{o_number}"
            formulas.append({
                'formula': formula_str,
                'composition': composition
            })

    return formulas