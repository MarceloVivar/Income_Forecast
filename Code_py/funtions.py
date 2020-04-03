import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import scipy.stats as st
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats import outliers_influence
from statsmodels.compat import lzip

#from descstats import MyPlot, Univa

import warnings
warnings.filterwarnings(action="ignore", module="sklearn",
                        message="^internal gelsd")

###############################################################
# Linear Regression Analysis
###############################################################


def linear_regression_analysis(linear_regression):
    """ Compute and plot a complete analysis of a linear regression computed with Stats Models.
    Args:
         linear_regression (Stats Models Results): the result obtained  with Stats Models.
    """

    # Data
    resid = linear_regression.resid_pearson.copy()
    resid_index = linear_regression.resid.index
    exog = linear_regression.model.exog
    endog = linear_regression.model.endog
    fitted_values = linear_regression.fittedvalues
    influences = outliers_influence.OLSInfluence(linear_regression)

    p = exog.shape[1]  # Number of features
    n = len(resid)  # Number of individuals

    # Paramètres
    color1 = "#3498db"
    color2 = "#e74c3c"

    ##############################################################################
    # Tests statistiques                                                         #
    ##############################################################################

    # Homoscédasticité - Test de Breusch-Pagan
    ##########################################

    names = ['Lagrande multiplier statistic',
             'p-value', 'f-value', 'f p-value']
    breusch_pagan = sm.stats.diagnostic.het_breuschpagan(resid, exog)
    print(lzip(names, breusch_pagan))

    # Test de normalité - Shapiro-Wilk
    ###################################

    print(f"Shapiro pvalue : {st.shapiro(resid)[1]}")

    ##############################################################################
    # Analyses de forme                                                          #
    ##############################################################################

    # Histogramme des résidus
    ##########################
    data = resid
    data_filter = data[data < 5]
    data_filter = data[data > -5]
    len_data = len(data)
    len_data_filter = len(data_filter)
    ratio = len_data_filter / len_data

    fig, ax = plt.subplots()
    plt.hist(data_filter, bins=20, color=color1)
    plt.xlabel("Residual values")
    plt.ylabel("Number of residuals")
    plt.title(f"Histogramme des résidus de -5 à 5 ({ratio:.2%})")

    # Normal distribution vs residuals (QQ Plot, droite de Henry)
    #############################################################
    data = pd.Series(resid).sort_values()
    len_data = len(data)

    normal = pd.Series(np.random.normal(size=len_data)).sort_values()
    fig, ax = plt.subplots()
    plt.scatter(data, normal, c=color1)
    plt.plot((-4, 4), (-4, 4), c=color2)
    plt.xlabel("Residuals")
    plt.ylabel("Normal distribution")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.title("Residuals vs Normal (QQ Plot)")

    # Plot
    plt.show()


def plot_sortie_acf(y_acf, y_len, pacf=False):
    "représentation de la sortie ACF"
    if pacf:
        y_acf = y_acf[1:]
    plt.figure(figsize=(14, 6))
    plt.bar(range(len(y_acf)), y_acf, width=0.1)
    plt.xlabel('lag')
    plt.ylabel('ACF')
    plt.axhline(y=0, color='black')
    plt.axhline(y=-1.96/np.sqrt(y_len), color='b',
                linestyle='--', linewidth=0.8)
    plt.axhline(y=1.96/np.sqrt(y_len), color='b',
                linestyle='--', linewidth=0.8)
    plt.ylim(-1, 1)
    plt.show()
    return
