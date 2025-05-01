
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define input/output variables (scale 1–5)
AE = ctrl.Antecedent(np.arange(1, 6, 1), 'AE')
LC = ctrl.Antecedent(np.arange(1, 6, 1), 'LC')
IC = ctrl.Antecedent(np.arange(1, 6, 1), 'IC')
IS = ctrl.Antecedent(np.arange(1, 6, 1), 'IS')
IQ = ctrl.Antecedent(np.arange(1, 6, 1), 'IQ')
RMF = ctrl.Antecedent(np.arange(1, 6, 1), 'RMF')
CFC = ctrl.Antecedent(np.arange(1, 6, 1), 'CFC')

# Final stage inputs (numerical values from previous outputs)
P = ctrl.Antecedent(np.arange(1, 6, 1), 'P')
I = ctrl.Antecedent(np.arange(1, 6, 1), 'I')
D = ctrl.Antecedent(np.arange(1, 6, 1), 'D')

# Intermediates and output
Probability = ctrl.Consequent(np.arange(1, 6, 1), 'Probability')
Impact = ctrl.Consequent(np.arange(1, 6, 1), 'Impact')
Detection = ctrl.Consequent(np.arange(1, 6, 1), 'Detection')
Risk = ctrl.Consequent(np.arange(1, 6, 1), 'Risk')


# Membership functions for variables with range 1–5
def define_mfs(var):
    var['Very Low'] = fuzz.trimf(var.universe, [1, 1, 2])
    var['Low'] = fuzz.trimf(var.universe, [1, 2, 3])
    var['Medium'] = fuzz.trimf(var.universe, [2, 3, 4])
    var['High'] = fuzz.trimf(var.universe, [3, 4, 5])
    var['Very High'] = fuzz.trimf(var.universe, [4, 5, 5])


for var in [AE, LC, IC, IS, IQ, RMF, CFC, P, I, D, Probability, Impact, Detection, Risk]:
    define_mfs(var)

# Rules for Probability
prob_rules = [
    # AE = Very Low
    ctrl.Rule(AE['Very Low'] & LC['Very Low'], Probability['Very High']),
    ctrl.Rule(AE['Very Low'] & LC['Low'], Probability['High']),
    ctrl.Rule(AE['Very Low'] & LC['Medium'], Probability['Medium']),
    ctrl.Rule(AE['Very Low'] & LC['High'], Probability['Medium']),
    ctrl.Rule(AE['Very Low'] & LC['Very High'], Probability['Low']),

    # AE = Low
    ctrl.Rule(AE['Low'] & LC['Very Low'], Probability['High']),
    ctrl.Rule(AE['Low'] & LC['Low'], Probability['High']),
    ctrl.Rule(AE['Low'] & LC['Medium'], Probability['Medium']),
    ctrl.Rule(AE['Low'] & LC['High'], Probability['Medium']),
    ctrl.Rule(AE['Low'] & LC['Very High'], Probability['Low']),

    # AE = Medium
    ctrl.Rule(AE['Medium'] & LC['Very Low'], Probability['High']),
    ctrl.Rule(AE['Medium'] & LC['Low'], Probability['Medium']),
    ctrl.Rule(AE['Medium'] & LC['Medium'], Probability['Medium']),
    ctrl.Rule(AE['Medium'] & LC['High'], Probability['Low']),
    ctrl.Rule(AE['Medium'] & LC['Very High'], Probability['Low']),

    # AE = High
    ctrl.Rule(AE['High'] & LC['Very Low'], Probability['Medium']),
    ctrl.Rule(AE['High'] & LC['Low'], Probability['Medium']),
    ctrl.Rule(AE['High'] & LC['Medium'], Probability['Low']),
    ctrl.Rule(AE['High'] & LC['High'], Probability['Low']),
    ctrl.Rule(AE['High'] & LC['Very High'], Probability['Very Low']),

    # AE = Very High
    ctrl.Rule(AE['Very High'] & LC['Very Low'], Probability['Low']),
    ctrl.Rule(AE['Very High'] & LC['Low'], Probability['Low']),
    ctrl.Rule(AE['Very High'] & LC['Medium'], Probability['Very Low']),
    ctrl.Rule(AE['Very High'] & LC['High'], Probability['Very Low']),
    ctrl.Rule(AE['Very High'] & LC['Very High'], Probability['Very Low'])
]


# Rules for Impact (explicit style)

impact_rules = [

    # IC = Very Low
    ctrl.Rule(IC['Very Low'] & IS['Very Low'] & IQ['Very Low'], Impact['Very Low']),
    ctrl.Rule(IC['Very Low'] & IS['Very Low'] & IQ['Low'], Impact['Low']),
    ctrl.Rule(IC['Very Low'] & IS['Very Low'] & IQ['Medium'], Impact['Medium']),
    ctrl.Rule(IC['Very Low'] & IS['Very Low'] & IQ['High'], Impact['Medium']),
    ctrl.Rule(IC['Very Low'] & IS['Very Low'] & IQ['Very High'], Impact['High']),
    ctrl.Rule(IC['Very Low'] & IS['Low'] & IQ['Very Low'], Impact['Low']),
    ctrl.Rule(IC['Very Low'] & IS['Low'] & IQ['Low'], Impact['Low']),
    ctrl.Rule(IC['Very Low'] & IS['Low'] & IQ['Medium'], Impact['Medium']),
    ctrl.Rule(IC['Very Low'] & IS['Low'] & IQ['High'], Impact['High']),
    ctrl.Rule(IC['Very Low'] & IS['Low'] & IQ['Very High'], Impact['High']),
    ctrl.Rule(IC['Very Low'] & IS['Medium'] & IQ['Very Low'], Impact['Medium']),
    ctrl.Rule(IC['Very Low'] & IS['Medium'] & IQ['Low'], Impact['Medium']),
    ctrl.Rule(IC['Very Low'] & IS['Medium'] & IQ['Medium'], Impact['Medium']),
    ctrl.Rule(IC['Very Low'] & IS['Medium'] & IQ['High'], Impact['High']),
    ctrl.Rule(IC['Very Low'] & IS['Medium'] & IQ['Very High'], Impact['High']),
    ctrl.Rule(IC['Very Low'] & IS['High'] & IQ['Very Low'], Impact['Medium']),
    ctrl.Rule(IC['Very Low'] & IS['High'] & IQ['Low'], Impact['High']),
    ctrl.Rule(IC['Very Low'] & IS['High'] & IQ['Medium'], Impact['High']),
    ctrl.Rule(IC['Very Low'] & IS['High'] & IQ['High'], Impact['High']),
    ctrl.Rule(IC['Very Low'] & IS['High'] & IQ['Very High'], Impact['High']),
    ctrl.Rule(IC['Very Low'] & IS['Very High'] & IQ['Very Low'], Impact['Medium']),
    ctrl.Rule(IC['Very Low'] & IS['Very High'] & IQ['Low'], Impact['High']),
    ctrl.Rule(IC['Very Low'] & IS['Very High'] & IQ['Medium'], Impact['High']),
    ctrl.Rule(IC['Very Low'] & IS['Very High'] & IQ['High'], Impact['High']),
    ctrl.Rule(IC['Very Low'] & IS['Very High'] & IQ['Very High'], Impact['Very High']),

    # IC = Low
    ctrl.Rule(IC['Low'] & IS['Very Low'] & IQ['Very Low'], Impact['Low']),
    ctrl.Rule(IC['Low'] & IS['Very Low'] & IQ['Low'], Impact['Low']),
    ctrl.Rule(IC['Low'] & IS['Very Low'] & IQ['Medium'], Impact['Medium']),
    ctrl.Rule(IC['Low'] & IS['Very Low'] & IQ['High'], Impact['Medium']),
    ctrl.Rule(IC['Low'] & IS['Very Low'] & IQ['Very High'], Impact['High']),
    ctrl.Rule(IC['Low'] & IS['Low'] & IQ['Very Low'], Impact['Medium']),
    ctrl.Rule(IC['Low'] & IS['Low'] & IQ['Low'], Impact['Medium']),
    ctrl.Rule(IC['Low'] & IS['Low'] & IQ['Medium'], Impact['Medium']),
    ctrl.Rule(IC['Low'] & IS['Low'] & IQ['High'], Impact['High']),
    ctrl.Rule(IC['Low'] & IS['Low'] & IQ['Very High'], Impact['High']),
    ctrl.Rule(IC['Low'] & IS['Medium'] & IQ['Very Low'], Impact['Medium']),
    ctrl.Rule(IC['Low'] & IS['Medium'] & IQ['Low'], Impact['Medium']),
    ctrl.Rule(IC['Low'] & IS['Medium'] & IQ['Medium'], Impact['High']),
    ctrl.Rule(IC['Low'] & IS['Medium'] & IQ['High'], Impact['High']),
    ctrl.Rule(IC['Low'] & IS['Medium'] & IQ['Very High'], Impact['High']),
    ctrl.Rule(IC['Low'] & IS['High'] & IQ['Very Low'], Impact['High']),
    ctrl.Rule(IC['Low'] & IS['High'] & IQ['Low'], Impact['High']),
    ctrl.Rule(IC['Low'] & IS['High'] & IQ['Medium'], Impact['High']),
    ctrl.Rule(IC['Low'] & IS['High'] & IQ['High'], Impact['High']),
    ctrl.Rule(IC['Low'] & IS['High'] & IQ['Very High'], Impact['Very High']),
    ctrl.Rule(IC['Low'] & IS['Very High'] & IQ['Very Low'], Impact['High']),
    ctrl.Rule(IC['Low'] & IS['Very High'] & IQ['Low'], Impact['High']),
    ctrl.Rule(IC['Low'] & IS['Very High'] & IQ['Medium'], Impact['High']),
    ctrl.Rule(IC['Low'] & IS['Very High'] & IQ['High'], Impact['Very High']),
    ctrl.Rule(IC['Low'] & IS['Very High'] & IQ['Very High'], Impact['Very High']),

    # IC = Medium
    ctrl.Rule(IC['Medium'] & IS['Very Low'] & IQ['Very Low'], Impact['Medium']),
    ctrl.Rule(IC['Medium'] & IS['Very Low'] & IQ['Low'], Impact['Medium']),
    ctrl.Rule(IC['Medium'] & IS['Very Low'] & IQ['Medium'], Impact['Medium']),
    ctrl.Rule(IC['Medium'] & IS['Very Low'] & IQ['High'], Impact['High']),
    ctrl.Rule(IC['Medium'] & IS['Very Low'] & IQ['Very High'], Impact['High']),

    ctrl.Rule(IC['Medium'] & IS['Low'] & IQ['Very Low'], Impact['Medium']),
    ctrl.Rule(IC['Medium'] & IS['Low'] & IQ['Low'], Impact['Medium']),
    ctrl.Rule(IC['Medium'] & IS['Low'] & IQ['Medium'], Impact['High']),
    ctrl.Rule(IC['Medium'] & IS['Low'] & IQ['High'], Impact['High']),
    ctrl.Rule(IC['Medium'] & IS['Low'] & IQ['Very High'], Impact['High']),

    ctrl.Rule(IC['Medium'] & IS['Medium'] & IQ['Very Low'], Impact['Medium']),
    ctrl.Rule(IC['Medium'] & IS['Medium'] & IQ['Low'], Impact['High']),
    ctrl.Rule(IC['Medium'] & IS['Medium'] & IQ['Medium'], Impact['High']),
    ctrl.Rule(IC['Medium'] & IS['Medium'] & IQ['High'], Impact['High']),
    ctrl.Rule(IC['Medium'] & IS['Medium'] & IQ['Very High'], Impact['Very High']),

    ctrl.Rule(IC['Medium'] & IS['High'] & IQ['Very Low'], Impact['High']),
    ctrl.Rule(IC['Medium'] & IS['High'] & IQ['Low'], Impact['High']),
    ctrl.Rule(IC['Medium'] & IS['High'] & IQ['Medium'], Impact['High']),
    ctrl.Rule(IC['Medium'] & IS['High'] & IQ['High'], Impact['Very High']),
    ctrl.Rule(IC['Medium'] & IS['High'] & IQ['Very High'], Impact['Very High']),

    ctrl.Rule(IC['Medium'] & IS['Very High'] & IQ['Very Low'], Impact['High']),
    ctrl.Rule(IC['Medium'] & IS['Very High'] & IQ['Low'], Impact['High']),
    ctrl.Rule(IC['Medium'] & IS['Very High'] & IQ['Medium'], Impact['Very High']),
    ctrl.Rule(IC['Medium'] & IS['Very High'] & IQ['High'], Impact['Very High']),
    ctrl.Rule(IC['Medium'] & IS['Very High'] & IQ['Very High'], Impact['Very High']),

    # IC = High
    ctrl.Rule(IC['High'] & IS['Very Low'] & IQ['Very Low'], Impact['High']),
    ctrl.Rule(IC['High'] & IS['Very Low'] & IQ['Low'], Impact['High']),
    ctrl.Rule(IC['High'] & IS['Very Low'] & IQ['Medium'], Impact['High']),
    ctrl.Rule(IC['High'] & IS['Very Low'] & IQ['High'], Impact['Very High']),
    ctrl.Rule(IC['High'] & IS['Very Low'] & IQ['Very High'], Impact['Very High']),

    ctrl.Rule(IC['High'] & IS['Low'] & IQ['Very Low'], Impact['High']),
    ctrl.Rule(IC['High'] & IS['Low'] & IQ['Low'], Impact['High']),
    ctrl.Rule(IC['High'] & IS['Low'] & IQ['Medium'], Impact['Very High']),
    ctrl.Rule(IC['High'] & IS['Low'] & IQ['High'], Impact['Very High']),
    ctrl.Rule(IC['High'] & IS['Low'] & IQ['Very High'], Impact['Very High']),

    ctrl.Rule(IC['High'] & IS['Medium'] & IQ['Very Low'], Impact['High']),
    ctrl.Rule(IC['High'] & IS['Medium'] & IQ['Low'], Impact['Very High']),
    ctrl.Rule(IC['High'] & IS['Medium'] & IQ['Medium'], Impact['Very High']),
    ctrl.Rule(IC['High'] & IS['Medium'] & IQ['High'], Impact['Very High']),
    ctrl.Rule(IC['High'] & IS['Medium'] & IQ['Very High'], Impact['Very High']),

    ctrl.Rule(IC['High'] & IS['High'] & IQ['Very Low'], Impact['Very High']),
    ctrl.Rule(IC['High'] & IS['High'] & IQ['Low'], Impact['Very High']),
    ctrl.Rule(IC['High'] & IS['High'] & IQ['Medium'], Impact['Very High']),
    ctrl.Rule(IC['High'] & IS['High'] & IQ['High'], Impact['Very High']),
    ctrl.Rule(IC['High'] & IS['High'] & IQ['Very High'], Impact['Very High']),

    ctrl.Rule(IC['High'] & IS['Very High'] & IQ['Very Low'], Impact['Very High']),
    ctrl.Rule(IC['High'] & IS['Very High'] & IQ['Low'], Impact['Very High']),
    ctrl.Rule(IC['High'] & IS['Very High'] & IQ['Medium'], Impact['Very High']),
    ctrl.Rule(IC['High'] & IS['Very High'] & IQ['High'], Impact['Very High']),
    ctrl.Rule(IC['High'] & IS['Very High'] & IQ['Very High'], Impact['Very High']),

    # IC = Very High
    ctrl.Rule(IC['Very High'] & IS['Very Low'] & IQ['Very Low'], Impact['High']),
    ctrl.Rule(IC['Very High'] & IS['Very Low'] & IQ['Low'], Impact['Very High']),
    ctrl.Rule(IC['Very High'] & IS['Very Low'] & IQ['Medium'], Impact['Very High']),
    ctrl.Rule(IC['Very High'] & IS['Very Low'] & IQ['High'], Impact['Very High']),
    ctrl.Rule(IC['Very High'] & IS['Very Low'] & IQ['Very High'], Impact['Very High']),

    ctrl.Rule(IC['Very High'] & IS['Low'] & IQ['Very Low'], Impact['Very High']),
    ctrl.Rule(IC['Very High'] & IS['Low'] & IQ['Low'], Impact['Very High']),
    ctrl.Rule(IC['Very High'] & IS['Low'] & IQ['Medium'], Impact['Very High']),
    ctrl.Rule(IC['Very High'] & IS['Low'] & IQ['High'], Impact['Very High']),
    ctrl.Rule(IC['Very High'] & IS['Low'] & IQ['Very High'], Impact['Very High']),

    ctrl.Rule(IC['Very High'] & IS['Medium'] & IQ['Very Low'], Impact['Very High']),
    ctrl.Rule(IC['Very High'] & IS['Medium'] & IQ['Low'], Impact['Very High']),
    ctrl.Rule(IC['Very High'] & IS['Medium'] & IQ['Medium'], Impact['Very High']),
    ctrl.Rule(IC['Very High'] & IS['Medium'] & IQ['High'], Impact['Very High']),
    ctrl.Rule(IC['Very High'] & IS['Medium'] & IQ['Very High'], Impact['Very High']),

    ctrl.Rule(IC['Very High'] & IS['High'] & IQ['Very Low'], Impact['Very High']),
    ctrl.Rule(IC['Very High'] & IS['High'] & IQ['Low'], Impact['Very High']),
    ctrl.Rule(IC['Very High'] & IS['High'] & IQ['Medium'], Impact['Very High']),
    ctrl.Rule(IC['Very High'] & IS['High'] & IQ['High'], Impact['Very High']),
    ctrl.Rule(IC['Very High'] & IS['High'] & IQ['Very High'], Impact['Very High']),

    ctrl.Rule(IC['Very High'] & IS['Very High'] & IQ['Very Low'], Impact['Very High']),
    ctrl.Rule(IC['Very High'] & IS['Very High'] & IQ['Low'], Impact['Very High']),
    ctrl.Rule(IC['Very High'] & IS['Very High'] & IQ['Medium'], Impact['Very High']),
    ctrl.Rule(IC['Very High'] & IS['Very High'] & IQ['High'], Impact['Very High']),
    ctrl.Rule(IC['Very High'] & IS['Very High'] & IQ['Very High'], Impact['Very High'])

    # (Continue rules for IC = Medium, High, Very High)
    # You can add the remaining 75 rules following this structure
]



# Detection = f(RMF, CFC)
detection_rules = [
    ctrl.Rule(RMF['Very Low'] & CFC['Very Low'], Detection['Very Low']),
    ctrl.Rule(RMF['Very Low'] & CFC['Low'], Detection['Very Low']),
    ctrl.Rule(RMF['Very Low'] & CFC['Medium'], Detection['Low']),
    ctrl.Rule(RMF['Very Low'] & CFC['High'], Detection['Low']),
    ctrl.Rule(RMF['Very Low'] & CFC['Very High'], Detection['Medium']),

    ctrl.Rule(RMF['Low'] & CFC['Very Low'], Detection['Very Low']),
    ctrl.Rule(RMF['Low'] & CFC['Low'], Detection['Low']),
    ctrl.Rule(RMF['Low'] & CFC['Medium'], Detection['Low']),
    ctrl.Rule(RMF['Low'] & CFC['High'], Detection['Medium']),
    ctrl.Rule(RMF['Low'] & CFC['Very High'], Detection['Medium']),

    ctrl.Rule(RMF['Medium'] & CFC['Very Low'], Detection['Low']),
    ctrl.Rule(RMF['Medium'] & CFC['Low'], Detection['Low']),
    ctrl.Rule(RMF['Medium'] & CFC['Medium'], Detection['Medium']),
    ctrl.Rule(RMF['Medium'] & CFC['High'], Detection['High']),
    ctrl.Rule(RMF['Medium'] & CFC['Very High'], Detection['High']),

    ctrl.Rule(RMF['High'] & CFC['Very Low'], Detection['Medium']),
    ctrl.Rule(RMF['High'] & CFC['Low'], Detection['Medium']),
    ctrl.Rule(RMF['High'] & CFC['Medium'], Detection['High']),
    ctrl.Rule(RMF['High'] & CFC['High'], Detection['High']),
    ctrl.Rule(RMF['High'] & CFC['Very High'], Detection['Very High']),

    ctrl.Rule(RMF['Very High'] & CFC['Very Low'], Detection['Medium']),
    ctrl.Rule(RMF['Very High'] & CFC['Low'], Detection['High']),
    ctrl.Rule(RMF['Very High'] & CFC['Medium'], Detection['High']),
    ctrl.Rule(RMF['Very High'] & CFC['High'], Detection['Very High']),
    ctrl.Rule(RMF['Very High'] & CFC['Very High'], Detection['Very High'])
]

risk_rules = [
    # P = Very Low
    ctrl.Rule(P['Very Low'] & I['Very Low'] & D['Very Low'], Risk['Very Low']),
    ctrl.Rule(P['Very Low'] & I['Very Low'] & D['Low'], Risk['Very Low']),
    ctrl.Rule(P['Very Low'] & I['Very Low'] & D['Medium'], Risk['Low']),
    ctrl.Rule(P['Very Low'] & I['Very Low'] & D['High'], Risk['Low']),
    ctrl.Rule(P['Very Low'] & I['Very Low'] & D['Very High'], Risk['Medium']),

    ctrl.Rule(P['Very Low'] & I['Low'] & D['Very Low'], Risk['Very Low']),
    ctrl.Rule(P['Very Low'] & I['Low'] & D['Low'], Risk['Low']),
    ctrl.Rule(P['Very Low'] & I['Low'] & D['Medium'], Risk['Low']),
    ctrl.Rule(P['Very Low'] & I['Low'] & D['High'], Risk['Medium']),
    ctrl.Rule(P['Very Low'] & I['Low'] & D['Very High'], Risk['Medium']),

    ctrl.Rule(P['Very Low'] & I['Medium'] & D['Very Low'], Risk['Low']),
    ctrl.Rule(P['Very Low'] & I['Medium'] & D['Low'], Risk['Medium']),
    ctrl.Rule(P['Very Low'] & I['Medium'] & D['Medium'], Risk['Medium']),
    ctrl.Rule(P['Very Low'] & I['Medium'] & D['High'], Risk['High']),
    ctrl.Rule(P['Very Low'] & I['Medium'] & D['Very High'], Risk['High']),

    ctrl.Rule(P['Very Low'] & I['High'] & D['Very Low'], Risk['Medium']),
    ctrl.Rule(P['Very Low'] & I['High'] & D['Low'], Risk['High']),
    ctrl.Rule(P['Very Low'] & I['High'] & D['Medium'], Risk['High']),
    ctrl.Rule(P['Very Low'] & I['High'] & D['High'], Risk['Very High']),
    ctrl.Rule(P['Very Low'] & I['High'] & D['Very High'], Risk['Very High']),

    ctrl.Rule(P['Very Low'] & I['Very High'] & D['Very Low'], Risk['High']),
    ctrl.Rule(P['Very Low'] & I['Very High'] & D['Low'], Risk['High']),
    ctrl.Rule(P['Very Low'] & I['Very High'] & D['Medium'], Risk['High']),
    ctrl.Rule(P['Very Low'] & I['Very High'] & D['High'], Risk['High']),
    ctrl.Rule(P['Very Low'] & I['Very High'] & D['Very High'], Risk['Very High']),

    # P = Low
    ctrl.Rule(P['Low'] & I['Very Low'] & D['Very Low'], Risk['Very Low']),
    ctrl.Rule(P['Low'] & I['Very Low'] & D['Low'], Risk['Low']),
    ctrl.Rule(P['Low'] & I['Very Low'] & D['Medium'], Risk['Low']),
    ctrl.Rule(P['Low'] & I['Very Low'] & D['High'], Risk['Medium']),
    ctrl.Rule(P['Low'] & I['Very Low'] & D['Very High'], Risk['Medium']),

    ctrl.Rule(P['Low'] & I['Low'] & D['Very Low'], Risk['Low']),
    ctrl.Rule(P['Low'] & I['Low'] & D['Low'], Risk['Low']),
    ctrl.Rule(P['Low'] & I['Low'] & D['Medium'], Risk['Medium']),
    ctrl.Rule(P['Low'] & I['Low'] & D['High'], Risk['Medium']),
    ctrl.Rule(P['Low'] & I['Low'] & D['Very High'], Risk['High']),

    ctrl.Rule(P['Low'] & I['Medium'] & D['Very Low'], Risk['Medium']),
    ctrl.Rule(P['Low'] & I['Medium'] & D['Low'], Risk['Medium']),
    ctrl.Rule(P['Low'] & I['Medium'] & D['Medium'], Risk['High']),
    ctrl.Rule(P['Low'] & I['Medium'] & D['High'], Risk['High']),
    ctrl.Rule(P['Low'] & I['Medium'] & D['Very High'], Risk['Very High']),

    ctrl.Rule(P['Low'] & I['High'] & D['Very Low'], Risk['High']),
    ctrl.Rule(P['Low'] & I['High'] & D['Low'], Risk['High']),
    ctrl.Rule(P['Low'] & I['High'] & D['Medium'], Risk['Very High']),
    ctrl.Rule(P['Low'] & I['High'] & D['High'], Risk['Very High']),
    ctrl.Rule(P['Low'] & I['High'] & D['Very High'], Risk['Very High']),

    ctrl.Rule(P['Low'] & I['Very High'] & D['Very Low'], Risk['High']),
    ctrl.Rule(P['Low'] & I['Very High'] & D['Low'], Risk['Very High']),
    ctrl.Rule(P['Low'] & I['Very High'] & D['Medium'], Risk['Very High']),
    ctrl.Rule(P['Low'] & I['Very High'] & D['High'], Risk['Very High']),
    ctrl.Rule(P['Low'] & I['Very High'] & D['Very High'], Risk['Very High']),

    # P = High
    ctrl.Rule(P['High'] & I['Very Low'] & D['Very Low'], Risk['Medium']),
    ctrl.Rule(P['High'] & I['Very Low'] & D['Low'], Risk['Medium']),
    ctrl.Rule(P['High'] & I['Very Low'] & D['Medium'], Risk['High']),
    ctrl.Rule(P['High'] & I['Very Low'] & D['High'], Risk['High']),
    ctrl.Rule(P['High'] & I['Very Low'] & D['Very High'], Risk['Very High']),
    ctrl.Rule(P['High'] & I['Low'] & D['Very Low'], Risk['Medium']),
    ctrl.Rule(P['High'] & I['Low'] & D['Low'], Risk['High']),
    ctrl.Rule(P['High'] & I['Low'] & D['Medium'], Risk['High']),
    ctrl.Rule(P['High'] & I['Low'] & D['High'], Risk['Very High']),
    ctrl.Rule(P['High'] & I['Low'] & D['Very High'], Risk['Very High']),
    ctrl.Rule(P['High'] & I['Medium'] & D['Very Low'], Risk['High']),
    ctrl.Rule(P['High'] & I['Medium'] & D['Low'], Risk['High']),
    ctrl.Rule(P['High'] & I['Medium'] & D['Medium'], Risk['Very High']),
    ctrl.Rule(P['High'] & I['Medium'] & D['High'], Risk['Very High']),
    ctrl.Rule(P['High'] & I['Medium'] & D['Very High'], Risk['Very High']),
    ctrl.Rule(P['High'] & I['High'] & D['Very Low'], Risk['Very High']),
    ctrl.Rule(P['High'] & I['High'] & D['Low'], Risk['Very High']),
    ctrl.Rule(P['High'] & I['High'] & D['Medium'], Risk['Very High']),
    ctrl.Rule(P['High'] & I['High'] & D['High'], Risk['Very High']),
    ctrl.Rule(P['High'] & I['High'] & D['Very High'], Risk['Very High']),
    ctrl.Rule(P['High'] & I['Very High'] & D['Very Low'], Risk['Very High']),
    ctrl.Rule(P['High'] & I['Very High'] & D['Low'], Risk['Very High']),
    ctrl.Rule(P['High'] & I['Very High'] & D['Medium'], Risk['Very High']),
    ctrl.Rule(P['High'] & I['Very High'] & D['High'], Risk['Very High']),
    ctrl.Rule(P['High'] & I['Very High'] & D['Very High'], Risk['Very High']),

# P = Very High
    ctrl.Rule(P['Very High'] & I['Very Low'] & D['Very Low'], Risk['High']),
    ctrl.Rule(P['Very High'] & I['Very Low'] & D['Low'], Risk['High']),
    ctrl.Rule(P['Very High'] & I['Very Low'] & D['Medium'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['Very Low'] & D['High'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['Very Low'] & D['Very High'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['Low'] & D['Very Low'], Risk['High']),
    ctrl.Rule(P['Very High'] & I['Low'] & D['Low'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['Low'] & D['Medium'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['Low'] & D['High'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['Low'] & D['Very High'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['Medium'] & D['Very Low'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['Medium'] & D['Low'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['Medium'] & D['Medium'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['Medium'] & D['High'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['Medium'] & D['Very High'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['High'] & D['Very Low'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['High'] & D['Low'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['High'] & D['Medium'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['High'] & D['High'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['High'] & D['Very High'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['Very High'] & D['Very Low'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['Very High'] & D['Low'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['Very High'] & D['Medium'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['Very High'] & D['High'], Risk['Very High']),
    ctrl.Rule(P['Very High'] & I['Very High'] & D['Very High'], Risk['Very High']),

    # P = Medium
    ctrl.Rule(P['Medium'] & I['Very Low'] & D['Very Low'], Risk['Medium']),
    ctrl.Rule(P['Medium'] & I['Very Low'] & D['Low'], Risk['Medium']),
    ctrl.Rule(P['Medium'] & I['Very Low'] & D['Medium'], Risk['High']),
    ctrl.Rule(P['Medium'] & I['Very Low'] & D['High'], Risk['High']),
    ctrl.Rule(P['Medium'] & I['Very Low'] & D['Very High'], Risk['High']),

    ctrl.Rule(P['Medium'] & I['Low'] & D['Very Low'], Risk['Medium']),
    ctrl.Rule(P['Medium'] & I['Low'] & D['Low'], Risk['High']),
    ctrl.Rule(P['Medium'] & I['Low'] & D['Medium'], Risk['High']),
    ctrl.Rule(P['Medium'] & I['Low'] & D['High'], Risk['High']),
    ctrl.Rule(P['Medium'] & I['Low'] & D['Very High'], Risk['Very High']),

    ctrl.Rule(P['Medium'] & I['Medium'] & D['Very Low'], Risk['High']),
    ctrl.Rule(P['Medium'] & I['Medium'] & D['Low'], Risk['High']),
    ctrl.Rule(P['Medium'] & I['Medium'] & D['Medium'], Risk['Very High']),
    ctrl.Rule(P['Medium'] & I['Medium'] & D['High'], Risk['Very High']),
    ctrl.Rule(P['Medium'] & I['Medium'] & D['Very High'], Risk['Very High']),

    ctrl.Rule(P['Medium'] & I['High'] & D['Very Low'], Risk['High']),
    ctrl.Rule(P['Medium'] & I['High'] & D['Low'], Risk['Very High']),
    ctrl.Rule(P['Medium'] & I['High'] & D['Medium'], Risk['Very High']),
    ctrl.Rule(P['Medium'] & I['High'] & D['High'], Risk['Very High']),
    ctrl.Rule(P['Medium'] & I['High'] & D['Very High'], Risk['Very High']),

    ctrl.Rule(P['Medium'] & I['Very High'] & D['Very Low'], Risk['Very High']),
    ctrl.Rule(P['Medium'] & I['Very High'] & D['Low'], Risk['Very High']),
    ctrl.Rule(P['Medium'] & I['Very High'] & D['Medium'], Risk['Very High']),
    ctrl.Rule(P['Medium'] & I['Very High'] & D['High'], Risk['Very High']),
    ctrl.Rule(P['Medium'] & I['Very High'] & D['Very High'], Risk['Very High']),

]


# Create control systems
prob_ctrl = ctrl.ControlSystem(prob_rules)
impact_ctrl = ctrl.ControlSystem(impact_rules)
detection_ctrl = ctrl.ControlSystem(detection_rules)
risk_ctrl = ctrl.ControlSystem(risk_rules)

# Simulation function
def compute_fuzzy_risk(ae_val, lc_val, ic_val, is_val, iq_val, rmf_val, cfc_val):
    prob_sim = ctrl.ControlSystemSimulation(prob_ctrl)
    prob_sim.input['AE'] = ae_val
    prob_sim.input['LC'] = lc_val
    prob_sim.compute()
    p_val = prob_sim.output['Probability']

    impact_sim = ctrl.ControlSystemSimulation(impact_ctrl)
    impact_sim.input['IC'] = ic_val
    impact_sim.input['IS'] = is_val
    impact_sim.input['IQ'] = iq_val
    impact_sim.compute()
    i_val = impact_sim.output['Impact']

    detection_sim = ctrl.ControlSystemSimulation(detection_ctrl)
    detection_sim.input['RMF'] = rmf_val
    detection_sim.input['CFC'] = cfc_val
    detection_sim.compute()
    d_val = detection_sim.output['Detection']

    risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)
    risk_sim.input['P'] = p_val
    risk_sim.input['I'] = i_val
    risk_sim.input['D'] = d_val
    risk_sim.compute()

    return round(risk_sim.output['Risk'], 2)

# Visualization function for final risk scores
def visualize_fuzzy_scores(df):
    import matplotlib.pyplot as plt

    if "Fuzzy Risk Number" not in df.columns:
        print("Fuzzy Risk Number column not found.")
        return

    df["Fuzzy Risk Number"] = df["Fuzzy Risk Number"].astype(float)

    # Bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(df["Risk Factor"], df["Fuzzy Risk Number"], color="skyblue")
    plt.xticks(rotation=90)
    plt.ylabel("Fuzzy Risk Number")
    plt.title("Fuzzy Risk Scores per Risk Factor")
    plt.tight_layout()
    plt.show()

    # Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(df["Fuzzy Risk Number"], bins=[0, 3.3, 6.6, 10], color="orange", edgecolor="black")
    plt.xticks([0, 3.3, 6.6, 10], ["Low", "Medium", "High"])
    plt.title("Distribution of Fuzzy Risk Levels")
    plt.xlabel("Risk Level")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def compute_fuzzy_probability(ae, lc):
    sim = ctrl.ControlSystemSimulation(prob_ctrl)
    sim.input['AE'] = ae
    sim.input['LC'] = lc
    sim.compute()
    return round(sim.output['Probability'], 2)

def compute_fuzzy_impact(ic, is_, iq):
    sim = ctrl.ControlSystemSimulation(impact_ctrl)
    sim.input['IC'] = ic
    sim.input['IS'] = is_
    sim.input['IQ'] = iq
    sim.compute()
    return round(sim.output['Impact'], 2)

def compute_fuzzy_detection(rmf, cfc):
    sim = ctrl.ControlSystemSimulation(detection_ctrl)
    sim.input['RMF'] = rmf
    sim.input['CFC'] = cfc
    sim.compute()
    return round(sim.output['Detection'], 2)


def show_surface_plot(sim, label1, label2, output_label):
    x_vals = np.arange(0, 11, 1)
    y_vals = np.arange(0, 11, 1)
    x, y = np.meshgrid(x_vals, y_vals)
    z = np.zeros_like(x, dtype=float)

    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            sim.input[label1] = x[i, j]
            sim.input[label2] = y[i, j]
            sim.compute()
            z[i, j] = sim.output[output_label]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    ax.set_zlabel(output_label)
    ax.set_title(f"Surface Viewer: {output_label} based on {label1} and {label2}")
    plt.tight_layout()
    plt.show()

    plt.close()

if __name__ == "__main__":
    # Sample values for testing (scale 1–5 as per your system)
    ae_val = 2   # Availability of Expertise
    lc_val = 4   # Level of Contingencies
    ic_val = 3   # Impact on Cost
    is_val = 4   # Impact on Schedule
    iq_val = 3   # Impact on Quality
    rmf_val = 2  # Risk Monitoring Frequency
    cfc_val = 4  # Cross-Functional Collaboration

    # Individual layer testing
    prob = compute_fuzzy_probability(ae_val, lc_val)
    impact = compute_fuzzy_impact(ic_val, is_val, iq_val)
    detect = compute_fuzzy_detection(rmf_val, cfc_val)
    risk = compute_fuzzy_risk(ae_val, lc_val, ic_val, is_val, iq_val, rmf_val, cfc_val)

    print("🧪 Fuzzy Layer Outputs")
    print(f"Probability Score: {prob}")
    print(f"Impact Score: {impact}")
    print(f"Detection Score: {detect}")
    print(f"Final Risk Score: {risk}")

