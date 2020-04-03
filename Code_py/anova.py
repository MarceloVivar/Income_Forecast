import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats







def add_eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    return aov


def add_omega_squared(aov):
    mse = aov['sum_sq'][-1]/aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse)
                       )/(sum(aov['sum_sq'])+mse)
    return aov


def calc_anova(x, y, data):

    k = len(pd.unique(data[x]))  # nombre de groupes
    N = len(data.values)  # taille de l'échantillon
    n = data.groupby(x).size()  # nb de valeurs par groupes

    #DF = Degré de liberté (Degree of Freedom)

    DFbetween = k - 1
    DFwithin = N - k
    DFtotal = N - 1

    moyenne_y = data[y].mean()

    classes = []

    for classe in data[x].unique():

        yi_classe = data[y][data[x] == classe]

        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean(),
                        'variance_classe': yi_classe.var(ddof=0)})

    SCT = sum([(yj-moyenne_y)**2 for yj in data[y]])

    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])

    SCR = sum([c['ni']*(c['variance_classe']) for c in classes])

    MSbetween = SCE/DFbetween
    MSwithin = SCR/DFwithin
    F_value = MSbetween/MSwithin
    p_value = stats.f.sf(F_value, DFbetween, DFwithin)

    resultat = dict({'SCE': round(SCE, 3), 'SCT': round(SCT, 3), 'SCR': round(
        SCR, 3), 'eta_squared': round(SCE/SCT, 3), 'Valeur_F': F_value, 'P-valeur': p_value})

    return resultat
