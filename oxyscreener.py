import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from src.engine import OxyScreener, generate_list
from src.processing import OxyData
from models.models import Classifier, Regressor


if __name__ == "__main__":

    print(f"")
    print(f"===== OXYSCREENER 1.0.0 =====")
    print(f"")
    print(f"Input Materials Project API key:")
    API_KEY = input()
    print(f"")

    processor = OxyData(API_KEY)
    print(f"=== Import data / Load data ===")
    print(f"1 - Import data (from Materials Project)")
    print(f"2 - Import & save data")
    print(f"3 - Load data (processed_data.pkl)")
    print(f"--- To use option 3, you must import & save data using option 2 beforehand ---")
    print(f"")
    options_process = input("Select option: ")
    if options_process == "1":
        print(f"")
        print(f"This may take a while...")
        print(f"")
        processor.process_data()
    elif options_process == "2":
        print(f"")
        print(f"This may take a while...")
        print(f"")
        processor.process_data()
        processor.save_to_file()
    elif options_process == "3":
        processor.load_data()
        print(f"")
    else:
        print(f"ERROR: Please input 1, 2 or 3!")
        sys.exit()


    print(f"=== ML MODELS ===")
    print(f"")
    print(f"--- Classification model ---")
    clf = Classifier(processor.df)
    clf.prep_data()
    print(f"")
    print(f"Initialize model / Load model")
    print(f"1 - Initialize model")
    print(f"2 - Initialize & save model")
    print(f"3 - Load model")
    print(f"")
    clf_opt = input(f"Select option: ")
    print(f"")
    if clf_opt == "1":
        clf.model()
        clf.train()
    elif clf_opt == "2":
        clf.model()
        clf.train()
        clf.save_model()
    elif clf_opt == "3":
        clf.load_model()
        clf.y_pred = clf.model_class.predict(clf.X_test)
    else:
        print(f"ERROR: Please input 1, 2 or 3!")
        sys.exit()


    print(f"")

    print(f"=== Show classification results? ===")
    print(f"1 - yes")
    print(f"2 - no")
    print(f"")
    c_y_n = input("Select option: ")
    if c_y_n == "1":
        clf.results()
        print(f"")
    elif c_y_n == "2":
        print(f"")
    else:
        print(f"ERROR: Please input 1 or 2!")
        sys.exit()

    print(f"--- Regression model ---")
    reg = Regressor(processor.df)
    reg.prep_data()
    print(f"")
    print(f"Initialize model / Load model")
    print(f"1 - Initialize model")
    print(f"2 - Initialize & save model")
    print(f"3 - Load model")
    print(f"")
    reg_opt = input(f"Select option: ")
    if reg_opt == "1":
        reg.model()
        reg.train()
    elif reg_opt == "2":
        reg.model()
        reg.train()
        reg.save_model()
    elif reg_opt == "3":
        reg.load_model()
        reg.yr_pred = reg.model_reg.predict(reg.Xr_test)
    else:
        print(f"ERROR: Please input 1, 2 or 3!")
        sys.exit()


    print(f"")

    print(f"=== Show regression results? ===")
    print(f"1 - yes")
    print(f"2 - no")
    print(f"")
    r_y_n = input(f"Select option: ")
    if r_y_n == "1":
        reg.results()
    elif r_y_n == "2":
        print(f"")
    else:
        print(f"ERROR: Please input 1 or 2!")
        sys.exit()


    # TESTER

    print(f"=== SUPPORT EVALUATION ===")
    print(f"")
    print(f"--- Support for 1 to 6 elements ---")
    print(f"")
    print(f"Input fixed atoms e.g. Ce 0.3, Zr 0.4, ...")
    fixed_input = input(f"Fixed atoms (hit Enter if none): ")

    fixed_dict = {}
    if fixed_input.strip():
        try:
            for item in fixed_input.split(','):
                parts = item.strip().split()
                if len(parts) == 2:
                    el, val = parts
                    fixed_dict[el] = float(val)
        except ValueError:
            print(f"ERROR: Invalid format for fixed atoms! Use 'Symbol Value'.")
            sys.exit()

    print(f"")
    print(f"Input atoms to scan e.g. La, Eu...")
    scan_input = input(f"Atoms to scan: ")

    scan_list = [s.strip() for s in scan_input.split(',') if s.strip()]
    if not scan_list:
        print(f"ERROR: Provide at least one element to scan!")
        sys.exit()

    print(f"")
    print(f"Input number of oxygen atoms")
    o_number = int(input("Number of oxygens: "))
    print(f"")
    print(f"Input step size")
    step = float(input(f"Step size: "))

    print(f"")
    print(f"Calculating energy & stability. This may take a while...")
    print(f"")
    form = generate_list(fixed_dict, scan_list, o_number, step)
    all_results = []


    conv = StrToComposition()
    magpie = ElementProperty.from_preset('magpie')

    for f in form:
        result = OxyScreener(f.get('formula'), clf.model_class, reg.model_reg)
        all_results.append(result)

    df_res = pd.DataFrame(all_results)



    # === STATISTICS ===

    # === SUMMARY STATISTICS ===
    print(f"")
    print(f"=== STATISTICS ===")
    print(f"")

    if not df_res.empty:
        df_stab = df_res[df_res['stability'] == 1].copy()
        if not df_stab.empty:
            print(f"--- Global Energy Stats [eV/atom] ---")
            stats = df_stab['energy'].describe()
            print(f"Mean: {stats['mean']:.4f}")
            print(f"Min (Best): {stats['min']:.4f}")
            print(f"Std: {stats['std']:.4f}")

            print(f"\n--- Top 5 Structures ---")
            top_5 = df_stab.sort_values(by=['confidence_stable', 'energy'], ascending=[False, True]).head(5)
            print(top_5[['formula', 'confidence_stable', 'energy']])


        else:
            print(f"")
            print("No stable materials found in the selected range.")
    else:
        print(f"")
        print("Generator failed to create any combinations.")
print(f"")

