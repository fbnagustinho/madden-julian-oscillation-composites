# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------------------------------------------------#
#           Script que gera analise descritiva da precipitação média sobre o Ceará a partir do dado .asc                 # 
#------------------------------------------------------------------------------------------------------------------------#

# Créditos do script

__author__ = ["Francisco Agustinho de Brito Neto"]
__email__ = "francisco.brito@funceme.br"
__credits__ = ["Bruno Dias", "Letícia Karyne"]

# Bibliotecas 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import locale

# abrindo o dados 

url = 'http://www5.funceme.br/web/storage/obs/dashes/data/daily/funceme/estado/asc/'

df = pd.read_csv(url + 'pr_daily_funceme_obs_19730101_20230614_thiessen_ceara.asc', 
                 names = ['Ano', 'Mês', 'Dia', 'Prec'], sep = ' ')

# Indexar no tempo o dado de precipitação 

df1 = pd.DataFrame({})

df1['time'] = pd.date_range(start = '1973-01-01', end = '2023-04-14')
df1['Prec'] = df['Prec']

# Set dataset nas datas de interesse

df1 = df1['Prec'].loc[(df1['time'] >= '1981-01-01') & (df1['time'] <= '2010-12-31')]

# Criando dataframe para trabalhar dado selecionado

df2 = pd.DataFrame({'time': pd.date_range(start = '1981-01-01', end = '2010-12-31'), 
                    'prec': np.array(df1)}).set_index('time')

# Calculando o acumulado mensal

prec_acummonth = df2.groupby(pd.Grouper(freq='M')).sum().reset_index()

locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
prec_acummonth['Meses'] = prec_acummonth['time'].dt.strftime("%b")

# Ciclo Sazonal 

prec_acummonth = prec_acummonth.set_index('time')

ciclo_sazonal = prec_acummonth.groupby(prec_acummonth.index.month)['prec'].mean()

# Descrição da chuva

total_anual = ciclo_sazonal.sum()

# percentual da chuva em cada mês 

chuva_percentual = (ciclo_sazonal / total_anual) * 100

# percentual acumulativo

chuvaperc_accum = chuva_percentual.cumsum()

# Figura do Boxplot 

fig, axs = plt.subplots(2, 1, dpi = 400)

bins = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez']

PROPS = {
    'boxprops':{'facecolor':'lightgray', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}}

sns.boxplot(x= "Meses", 
                    y= "prec", 
                    data=prec_acummonth,
                    #showmeans=True,
                    showfliers = False,
                    ax=axs[0],
                    #meanprops={"marker":"o",
                    #           "markerfacecolor":"black", 
                    #           "markeredgecolor":"black",
                    #           "markersize":"6",
                    #           "label": "Média"},
                    **PROPS)

# sns.stripplot(x="Meses", 
#               y="prec", 
#               data=prec_acummonth,
#               jitter=True, 
#               linewidth=1.0, 
#               color = 'gray', 
#               marker = '*', 
#               label = 'Observações', 
#               ax=axs[0])

campo = sns.pointplot(x="Meses", 
                      y="prec", 
                      data=prec_acummonth, 
                      linestyles='--', 
                      scale=0.6, 
                      color='k', 
                      errwidth=0, 
                      capsize=0, 
                      ax=axs[0], 
                      label = 'Climatological Mean (1981 - 2010)')

handles, labels = campo.get_legend_handles_labels()

a = np.array([labels[0], labels[-1]])
b = np.array([handles[0], handles[-1]])

axs[0].legend(b[:1], a[:1], fontsize = 8)

axs[0].set_ylabel("mm", size=10)
axs[0].set_xlabel("Months", size=10)
axs[0].set_ylim(0,500)
#axs[0].set_xticks(size = 15)
#axs[0].set_yticks(size = 15)
axs[0].set_title("(a) Seasonal Precipitation Cycle", size=9, loc = 'left', weight = 'bold')

# Figura b

rects = axs[1].bar(bins, np.array(chuva_percentual).round(decimals=1), color = 'gray', label = 'Percentual Mensal')
campo1 = axs[1].plot(bins, np.array(chuvaperc_accum), ls = '--', marker = 'o', color = 'black', label = 'Percentual Acumulado')

for x,y, z in zip(bins[0:], np.array(chuvaperc_accum[0:]).round(decimals=1),  np.array(chuvaperc_accum[0:]).round(decimals=1)):
            
    label = np.array(z)
            
    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,8), # distance from text to points (x,y)
                 ha='center', # horizontal alignment can be left, right or center
                 fontsize = 8) 

#axs[1].bar_label(rects, padding=2.3, fontsize = 7)
axs[1].set_ylim(0,120)
axs[1].set_xticklabels(['JAN', 'FEV', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])
axs[1].set_yticklabels(np.arange(0,120,20))
axs[1].set_title("(b) Monthly Rainfall Percentage Contribution", size=9, loc = 'left', weight = 'bold')
axs[1].set_ylabel("%", size=10)
axs[1].set_xlabel("Meses", size=10)

axs[1].legend(loc='upper left', fontsize = 8)

plt.subplots_adjust(top=0.95,
                    bottom=0.09,
                    left=0.1,
                    right=0.97,
                    hspace=0.4,
                    wspace=0.2)

plt.savefig('./figuras/caract_chuva_mensal_inglês.png', dpi = 600)

#plt.show()