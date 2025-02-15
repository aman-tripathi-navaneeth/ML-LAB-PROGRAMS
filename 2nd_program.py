import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv('C:\data.csv')
concepts = np.array(data.iloc[:, :-1])
print("\nInstances are \n", concepts)
target = np.array(data.iloc[:, -1])
print("\nTarget values are:", target)
                                         
def train(concepts, target):
    specific_h = concepts[0].copy()
    print("\nInitialize specific hypothesis:", specific_h)
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print("\nInitialize general hypothesis:", general_h)

    for i, h in enumerate(concepts):
        if target[i] == "yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        if target[i] == "no":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        print("\nSteps of Candidate Elimination Algorithm", i+1)
        print(specific_h)
        print(general_h)

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])

    return specific_h, general_h

s_final, g_final = train(concepts, target)

print("\nFinal Specific Hypothesis:", s_final)
print("\nFinal General Hypothesis:", g_final)   
