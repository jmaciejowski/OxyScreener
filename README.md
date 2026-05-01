# OxyScreener - ML-powered catalyst support discovery

## OxyScreener requires API Key from Materials Project!

## About OxyScreener
OxyScreener is a program dedicated to material scientists 
for the analysis and discovery of heterogeneous catalyst supports, 
including High Entropy Oxides (HEO). 
These materials are critical for their ability to store and release 
oxygen effectively during oxidation reactions. 
The program is designed to determine the general thermodynamic 
stability of various oxide supports and predict their formation 
energy per atom.

The program utilizes a database of over 80,000 oxides with metallic compositions 
ranging from 1 to 6 elements, sourced
from Materials Project. The data is processed through a cascade 
Machine Learning (ML) model: 
1) Classification Model - determines 
stability of catalyst supports based on initial threshold
of 0.05 eV/atom of energy above hull)
2) Regression Model - predicts formation energy per atom [eV/atom]
of analyzed supports. 

While **formation energy per atom** does not directly measure a 
support's oxygen bonding ability, it serves as a powerful proxy. 
Due to scaling relations, a support's ability to bond oxygen is 
closely correlated with this thermodynamic feature [1].

The initial settings of ML model allows it to reach f1-score of
~0.82 for predicting both stable & unstable supports and R2 ~ 0.987
with MAE ~ 0.049 eV/atom for predicting.

## How to use OxyScreener

OxyScreener requires the user to have Materials Project API KEY
as it loads a data from their database. Upon startup, the user 
will be prompted to provide this key to access the database.

### Import data / Load data

Input a number between 1-3 in the command window to:
1) **Import data (from Materials Project)** - Fetches data from Materials Project. (Note: This process is time-consuming).
2) **Import & save data** - Fetches data and saves it locally for faster access in future sessions.
3) **Load data (processed_data.pkl)** - Loads previously saved data.

### Classification & Regression models
1) **Initialize model** - Re-initializes the model from scratch.
2) **Initialize & save model** - Re-initializes the model from scratch & saves it locally.
3) **Load model** - Loads pre-trained models. 
OxyScreener includes saved models for immediate use.

### Results visualization

Users can choose to visualize model performance metrics:
1) Classification model:
   1) Feature importance plot
   2) Model performance (f1-score)
   3) Confusion matrix


2) Regression model:
   1) predicted y vs. true y plot
   2) R2 value
   3) MAE value

### Support evaluation

This section allows users to perform High-Throughput Screening of custom compositions 
using a "budget-based" concentration logic:
1) **Fixed atoms** - Optional. Freeze specific concentrations (e.g. 
```Ce 0.3, Zr 0.2```). Press 'Enter' to skip.
2) **Atoms to scan** - Elements that will dynamically fill 
the remaining concentration budget (e.g.```La, Eu```).
3) **Number of oxygen atoms** - Integer value representing the oxygen stoichiometry.
4) **Step** - A value between 0.0 and 1.0 (e.g.```0.1```) defining the scan density.

**Example:** Input ```Ce 0.3``` as fixed atom, ```La, Eu``` as atoms to scan, ```2``` oxygen atoms and step size ```0.1```. As a result, OxyScreener will
output all possible combinations of 
Ce<sub>0.3</sub>La<sub>x</sub>Eu<sub>0.7-x</sub> where x = 0.1, 0.2, 0.3 ,...

Single Structure Testing: To test a specific formula like 
Zr<sub>0.4</sub>Gd<sub>0.4</sub>Sm<sub>0.2</sub>O<sub>2</sub>, input Zr 0.4, Gd 0.4 as 
fixed atoms and Sm as the scan atom with a step size of 0.2

Keep in mind that results such as Ce<sub>1.0</sub>Eu<sub>0.0</sub>O<sub>2</sub>
or Ce<sub>0.0</sub>Eu<sub>1.0</sub>O<sub>2</sub> correspond to
CeO<sub>2</sub> and EuO<sub>2</sub>

The program will output stability prediction as well as the confidence
of stability prediction and energy per atom:

~~~
=== Ce1.0La0.0O2 ===
Stability: 1
Confidence: 94.29%
Energy: -3.87
~~~

Lastly, the statistics will be outputted:

~~~
=== STATISTICS ===

--- Global Energy Stats [eV/atom] ---
Mean: -3.7091
Min (Best): -3.8423
Std: 0.0630

--- Top 5 Structures ---
                 formula  confidence_stable    energy
0         Ce0.30Sm0.70O2           0.795214 -3.726901
21  Ce0.30La0.30Sm0.40O2           0.706904 -3.777810
8   Ce0.30La0.10Sm0.60O2           0.697911 -3.744653
30  Ce0.30La0.50Sm0.20O2           0.691962 -3.802219
32  Ce0.30La0.50Eu0.20O2           0.690873 -3.684041
~~~

## Limitations 

1) The program doesn't predict direct supports 
ability to bond oxygen [1].
2) OxyScreener is unable to differentiate structures that exhibit
polymorphism (same formula, different structure). 
Therefore, before ML module the program leaves only one
of the polymorphic structures with the lowest **energy above hull**
(since it is the most stable one) for each of the oxides that exhibit 
polymorphism.
3) The program doesn't output the crystallographic data of the supports.


# Literature:
[1] Dickens, Colin F., et al. "An electronic structure descriptor for oxygen reactivity at metal and metal-oxide surfaces." Surface Science 681 (2019): 122-129.