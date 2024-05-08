# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------------------------------------------------#
#                         Script que calcula os composto de OLR de acordo com as fases da OMJ                            #
#                Os composto serão em função das datas com fases ativas e neutralidade dos eventos de OMJ                #  
#                        Script no final salva as figuras com o composto no diretório pré-definido                       #     
#------------------------------------------------------------------------------------------------------------------------#

# Créditos do script

__author__ = ["Francisco Agustinho de Brito Neto"]
__email__ = "acer@funceme.br"
__credits__ = ["Bruno Dias"]

# Bibliotecas 

import numpy as np
import pandas as pd 
import xarray as xr
from tqdm import tqdm
from scipy import stats
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.shapereader as shpreader 
from matplotlib.patches import Path, PathPatch
from cartopy.util import add_cyclic_point

# Abrindo o dado de chuva filtrado

path = '/media/francisco.brito/FUNCEME/trabalhos/dataset/field_trat/'

ds_olr = xr.open_dataset(path + 'olrA_1979_2022_lanczos20200_index_clim8110.nc')

# Ajuste a longitude para começar em 20 leste

ds_olr.coords['lon'] = (ds_olr.coords['lon'] + 180) % 360 - 180
ds_olr = ds_olr.sortby(ds_olr.lon)

# Defina a longitude desejada para o centro (180 graus)
center_longitude = 180

# Crie uma nova grade de longitude centrada em 180 graus
new_lon = np.linspace(0, 360, len(ds_olr.lon), endpoint=False) - center_longitude

# Atribua a nova grade de longitude aos dados
ds_olr['lon'] = new_lon

# Abrindo os dados dos eventos de OMJ

df_mjo = pd.read_csv('/media/francisco.brito/FUNCEME/trabalhos/dataset/dataset_events/' \
                     'st_events_mjo_19812022.csv')

# Set datas para realização do composites 

df_mjo['time'] = pd.to_datetime(df_mjo['time'])

# Separando as data por fases

fase0_data = {}
fase1_data = {}
fase2_data = {}
fase3_data = {}
fase4_data = {}
fase5_data = {}
fase6_data = {}
fase7_data = {}
fase8_data = {}

for n in tqdm(range(0,len(df_mjo['fases']))):

    if df_mjo['fases'][n] == 0:
        
        fase0_data[n] = pd.to_datetime(df_mjo['time'][n])

    if df_mjo['fases'][n] == 1:
        
        fase1_data[n] = pd.to_datetime(df_mjo['time'][n])

    if df_mjo['fases'][n] == 2:
        
        fase2_data[n] = pd.to_datetime(df_mjo['time'][n])    

    if df_mjo['fases'][n] == 3:
        
        fase3_data[n] = pd.to_datetime(df_mjo['time'][n]) 

    if df_mjo['fases'][n] == 4:
        
        fase4_data[n] = pd.to_datetime(df_mjo['time'][n]) 

    if df_mjo['fases'][n] == 5:
        
        fase5_data[n] = pd.to_datetime(df_mjo['time'][n]) 

    if df_mjo['fases'][n] == 6:
        
        fase6_data[n] = pd.to_datetime(df_mjo['time'][n]) 

    if df_mjo['fases'][n] == 7:
        
        fase7_data[n] = pd.to_datetime(df_mjo['time'][n]) 

    if df_mjo['fases'][n] == 8:
        
        fase8_data[n] = pd.to_datetime(df_mjo['time'][n]) 
        
fase0data_list = pd.to_datetime(list(fase0_data.values()))
fase1data_list = pd.to_datetime(list(fase1_data.values()))
fase2data_list = pd.to_datetime(list(fase2_data.values()))
fase3data_list = pd.to_datetime(list(fase3_data.values()))
fase4data_list = pd.to_datetime(list(fase4_data.values()))
fase5data_list = pd.to_datetime(list(fase5_data.values()))
fase6data_list = pd.to_datetime(list(fase6_data.values()))
fase7data_list = pd.to_datetime(list(fase7_data.values()))
fase8data_list = pd.to_datetime(list(fase8_data.values()))

# Contando quantos eventos de MJO teve em cada ponto de grade

fase0_count = ds_olr['uwnd_lanczos'].sel(time=np.array(fase0data_list)).groupby('time.month').count('time')
fase1_count = ds_olr['uwnd_lanczos'].sel(time=np.array(fase1data_list)).groupby('time.month').count('time')
fase2_count = ds_olr['uwnd_lanczos'].sel(time=np.array(fase2data_list)).groupby('time.month').count('time')
fase3_count = ds_olr['uwnd_lanczos'].sel(time=np.array(fase3data_list)).groupby('time.month').count('time')
fase4_count = ds_olr['uwnd_lanczos'].sel(time=np.array(fase4data_list)).groupby('time.month').count('time')
fase5_count = ds_olr['uwnd_lanczos'].sel(time=np.array(fase5data_list)).groupby('time.month').count('time')
fase6_count = ds_olr['uwnd_lanczos'].sel(time=np.array(fase6data_list)).groupby('time.month').count('time')
fase7_count = ds_olr['uwnd_lanczos'].sel(time=np.array(fase7data_list)).groupby('time.month').count('time')
fase8_count = ds_olr['uwnd_lanczos'].sel(time=np.array(fase8data_list)).groupby('time.month').count('time')

# período analisado

#lista_periodo = [1, 2, 3, 4, 5, 6, 7, 12] # DJFMAMJJ
#lista_periodo = [1, 12] # DJ
#lista_periodo = [6, 7] # JJ
#lista_periodo = [2, 3, 4, 5] # FMAM
lista_periodo = [1, 2, 3, 4, 5, 12] # DJFMAM
#lista_periodo = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # Year

fase0_FMAMcount = fase0_count.sel(month = lista_periodo).sum('month')
fase1_FMAMcount = fase1_count.sel(month = lista_periodo).sum('month')
fase2_FMAMcount = fase2_count.sel(month = lista_periodo).sum('month')
fase3_FMAMcount = fase3_count.sel(month = lista_periodo).sum('month')
fase4_FMAMcount = fase4_count.sel(month = lista_periodo).sum('month')
fase5_FMAMcount = fase5_count.sel(month = lista_periodo).sum('month')
fase6_FMAMcount = fase6_count.sel(month = lista_periodo).sum('month')
fase7_FMAMcount = fase7_count.sel(month = lista_periodo).sum('month')
fase8_FMAMcount = fase8_count.sel(month = lista_periodo).sum('month')

total = fase0_FMAMcount + fase1_FMAMcount + fase2_FMAMcount + fase3_FMAMcount + fase4_FMAMcount + fase5_FMAMcount + fase6_FMAMcount + fase7_FMAMcount + fase8_FMAMcount

# pesos para média poderada

fase0p = np.array(fase0_FMAMcount)/np.array(total)
fase1p = np.array(fase1_FMAMcount)/np.array(total)
fase2p = np.array(fase2_FMAMcount)/np.array(total)
fase3p = np.array(fase3_FMAMcount)/np.array(total)
fase4p = np.array(fase4_FMAMcount)/np.array(total)
fase5p = np.array(fase5_FMAMcount)/np.array(total)
fase6p = np.array(fase6_FMAMcount)/np.array(total)
fase7p = np.array(fase7_FMAMcount)/np.array(total)
fase8p = np.array(fase8_FMAMcount)/np.array(total)

# Isolado o período por meses

fase0 = ds_olr['uwnd_lanczos'].sel(time=np.array(fase0data_list)).groupby('time.month').mean('time')
fase1 = ds_olr['uwnd_lanczos'].sel(time=np.array(fase1data_list)).groupby('time.month').mean('time')
fase2 = ds_olr['uwnd_lanczos'].sel(time=np.array(fase2data_list)).groupby('time.month').mean('time')
fase3 = ds_olr['uwnd_lanczos'].sel(time=np.array(fase3data_list)).groupby('time.month').mean('time')
fase4 = ds_olr['uwnd_lanczos'].sel(time=np.array(fase4data_list)).groupby('time.month').mean('time')
fase5 = ds_olr['uwnd_lanczos'].sel(time=np.array(fase5data_list)).groupby('time.month').mean('time')
fase6 = ds_olr['uwnd_lanczos'].sel(time=np.array(fase6data_list)).groupby('time.month').mean('time')
fase7 = ds_olr['uwnd_lanczos'].sel(time=np.array(fase7data_list)).groupby('time.month').mean('time')
fase8 = ds_olr['uwnd_lanczos'].sel(time=np.array(fase8data_list)).groupby('time.month').mean('time')

fase0_FMAM = fase0.sel(month = lista_periodo).mean('month')
fase1_FMAM = fase1.sel(month = lista_periodo).mean('month')

#fase1_FMAM = np.where(fase1_FMAM > 5.0, fase1_FMAM, np.where(fase1_FMAM < -5.0, fase1_FMAM, np.nan))

fase2_FMAM = fase2.sel(month = lista_periodo).mean('month')
fase3_FMAM = fase3.sel(month = lista_periodo).mean('month')
fase4_FMAM = fase4.sel(month = lista_periodo).mean('month')
fase5_FMAM = fase5.sel(month = lista_periodo).mean('month')
fase6_FMAM = fase6.sel(month = lista_periodo).mean('month')
fase7_FMAM = fase7.sel(month = lista_periodo).mean('month')
fase8_FMAM = fase8.sel(month = lista_periodo).mean('month')

fase0_FMAMstd = fase0.sel(month = lista_periodo).std('month')
fase1_FMAMstd = fase1.sel(month = lista_periodo).std('month')
fase2_FMAMstd = fase2.sel(month = lista_periodo).std('month')
fase3_FMAMstd = fase3.sel(month = lista_periodo).std('month')
fase4_FMAMstd = fase4.sel(month = lista_periodo).std('month')
fase5_FMAMstd = fase5.sel(month = lista_periodo).std('month')
fase6_FMAMstd = fase6.sel(month = lista_periodo).std('month')
fase7_FMAMstd = fase7.sel(month = lista_periodo).std('month')
fase8_FMAMstd = fase8.sel(month = lista_periodo).std('month')

#------------------------------------------------------------------------------#

# Lista de nomes das fases
#fase_names = ['Fase 1', 'Fase 2', 'Fase 3', 'Fase 4', 'Fase 5', 'Fase 6', 'Fase 7', 'Fase 8']
fase_names = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5', 'Phase 6', 'Phase 7', 'Phase 8']
# Figuras espaciais dos trimestres
crs = ccrs.PlateCarree(central_longitude=180)

fig, axs = plt.subplots(8, 1, dpi=600, subplot_kw={'projection': crs})

lon = ds_olr['lon']
lat = ds_olr['lat']
lons, lats = np.meshgrid(lon, lat)

#bounds = np.arange(-12, 15, 3)

bounds = [-10, -7.5, -5.0, -2.0, 0, 2.0, 5.0, 7.5, 10]

#cmap = 'bwr'

colors = ['#1464d2', '#2882f0', '#50a5f5', '#96d2fa', '#ffffff', '#ffffff', '#ffc03c', '#ff6000', '#e11400', '#a50000']

# for i, (ax, fase, fase_name) in enumerate(zip(axs, [fase1_FMAM, fase2_FMAM, fase3_FMAM, fase4_FMAM, fase5_FMAM, fase6_FMAM, fase7_FMAM, fase8_FMAM],
#                                                    ['Anomalia de ROL Filtrado', '', '', '', '', '', '', '', ''])):

for i, (ax, fase, fase_name) in enumerate(zip(axs, [fase1_FMAM, fase2_FMAM, fase3_FMAM, fase4_FMAM, fase5_FMAM, fase6_FMAM, fase7_FMAM, fase8_FMAM],
                                                   fase_names)):
    
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)

    fase, lon1 = add_cyclic_point(fase, coord=lon)

    #campo = ax.contourf(lon1, lat, fase, levels=bounds, colors = colors, transform=ccrs.PlateCarree(), extend='both')

    if i == 7:

        campo = ax.contourf(lon1, lat, fase, levels=bounds, colors=colors, transform=ccrs.PlateCarree(), extend='both')

        ax.set_xticks(np.arange(-180, 180, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-15., 25, 15), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.xaxis.set_tick_params(labelsize = 7)
        ax.yaxis.set_tick_params(labelsize = 7)

        # Adicionando rótulo à direita
        ax.text(1.01, 0.5, fase_name, rotation=90, transform=ax.transAxes, fontsize=6, weight='bold', va='center')

    else:

        campo = ax.contourf(lon1, lat, fase, levels=bounds, colors=colors, transform=ccrs.PlateCarree(), extend='both')
        ax.set_yticks(np.arange(-15.,25,15), crs=ccrs.PlateCarree())
        lat_formatter = LatitudeFormatter()
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.yaxis.set_tick_params(labelsize = 7)

        # Adicionando rótulo à direita
        ax.text(1.01, 0.5, fase_name, rotation=90, transform=ax.transAxes, fontsize=7, weight='bold', va='center')

# Adicionando título centralizado na parte superior
#fig.suptitle('Anomalia de ROL Filtrado - Compostos por Fase da OMJ', fontsize=8, weight='bold')
fig.suptitle('Composites of Filtered OLR anomalies - DJFMAM', 
             fontsize = 9, 
             weight='bold')

# Adicionando barra de cores

cax = plt.axes([0.87, 0.1, 0.013, 0.8])

fig.colorbar(campo, cax = cax, ticks = bounds, label='w/m²')

plt.subplots_adjust(top=0.94,
                    bottom=0.06,
                    left=0.06,
                    right=0.83,
                    hspace=0.2,
                    wspace=0.2)
#plt.tight_layout()

plt.savefig('./figuras/olr_filtrado_composite_djfmam.png', dpi = 600)
#plt.show()