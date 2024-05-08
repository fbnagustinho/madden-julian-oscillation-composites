# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------------------------------------------------#
#       Script que calcula os composto da precipitação oriundo da Krigagem do CE com a significância estatística         #
#                Os composto serão em função das datas com fases ativas e neutralidade dos eventos de OMJ                #  
#                        Script no final salva as figuras com o composto no diretório pré-definido                       #     
#------------------------------------------------------------------------------------------------------------------------#

# Créditos do script

__author__ = ["Francisco Agustinho de Brito Neto"]
__email__ = "francisco.brito@funceme.br"
__credits__ = ["Bruno Dias", "Letícia Karyne"]

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

# Abrindo o dado de chuva filtrado 

ds_prec = xr.open_dataset('./dados/prec_FFTanom2090_scipy_CE_1981_2022_clim8122_daily_krigagem.nc')

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

fase0_count = ds_prec['precip'].sel(time=np.array(fase0data_list)).groupby('time.month').count('time')
fase1_count = ds_prec['precip'].sel(time=np.array(fase1data_list)).groupby('time.month').count('time')
fase2_count = ds_prec['precip'].sel(time=np.array(fase2data_list)).groupby('time.month').count('time')
fase3_count = ds_prec['precip'].sel(time=np.array(fase3data_list)).groupby('time.month').count('time')
fase4_count = ds_prec['precip'].sel(time=np.array(fase4data_list)).groupby('time.month').count('time')
fase5_count = ds_prec['precip'].sel(time=np.array(fase5data_list)).groupby('time.month').count('time')
fase6_count = ds_prec['precip'].sel(time=np.array(fase6data_list)).groupby('time.month').count('time')
fase7_count = ds_prec['precip'].sel(time=np.array(fase7data_list)).groupby('time.month').count('time')
fase8_count = ds_prec['precip'].sel(time=np.array(fase8data_list)).groupby('time.month').count('time')

# período analisado

#lista_periodo = [1, 2, 3, 4, 5, 6, 7, 12] # DJFMAMJJ
#lista_periodo = [1, 12] # DJ
#lista_periodo = [6, 7] # JJ
#lista_periodo = [2, 3, 4, 5] # FMAM
lista_periodo = [1, 2, 3, 4, 5, 12] # DJFMAM


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

fase0 = ds_prec['precip'].sel(time=np.array(fase0data_list)).groupby('time.month').mean('time')
fase1 = ds_prec['precip'].sel(time=np.array(fase1data_list)).groupby('time.month').mean('time')
fase2 = ds_prec['precip'].sel(time=np.array(fase2data_list)).groupby('time.month').mean('time')
fase3 = ds_prec['precip'].sel(time=np.array(fase3data_list)).groupby('time.month').mean('time')
fase4 = ds_prec['precip'].sel(time=np.array(fase4data_list)).groupby('time.month').mean('time')
fase5 = ds_prec['precip'].sel(time=np.array(fase5data_list)).groupby('time.month').mean('time')
fase6 = ds_prec['precip'].sel(time=np.array(fase6data_list)).groupby('time.month').mean('time')
fase7 = ds_prec['precip'].sel(time=np.array(fase7data_list)).groupby('time.month').mean('time')
fase8 = ds_prec['precip'].sel(time=np.array(fase8data_list)).groupby('time.month').mean('time')

# FMAM

# fase0_FMAM = fase0.sel(month =slice(2,5)).mean('month') * fase0p
# fase1_FMAM = fase1.sel(month =slice(2,5)).mean('month') * fase1p
# fase2_FMAM = fase2.sel(month =slice(2,5)).mean('month') * fase2p
# fase3_FMAM = fase3.sel(month =slice(2,5)).mean('month') * fase3p
# fase4_FMAM = fase4.sel(month =slice(2,5)).mean('month') * fase4p
# fase5_FMAM = fase5.sel(month =slice(2,5)).mean('month') * fase5p
# fase6_FMAM = fase6.sel(month =slice(2,5)).mean('month') * fase6p
# fase7_FMAM = fase7.sel(month =slice(2,5)).mean('month') * fase7p
# fase8_FMAM = fase8.sel(month =slice(2,5)).mean('month') * fase8p

fase0_FMAM = fase0.sel(month = lista_periodo).mean('month')
fase1_FMAM = fase1.sel(month = lista_periodo).mean('month')
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

# Analise de significância estatística

# Define o nível de significancia (alfa)

alfa = 0.01

# Calcula o valor t-tabulado

t_tabulado1 = stats.t.ppf(1 - alfa/2, fase1_FMAMcount-1)
t_tabulado2 = stats.t.ppf(1 - alfa/2, fase2_FMAMcount-1)
t_tabulado3 = stats.t.ppf(1 - alfa/2, fase3_FMAMcount-1)
t_tabulado4 = stats.t.ppf(1 - alfa/2, fase4_FMAMcount-1)
t_tabulado5 = stats.t.ppf(1 - alfa/2, fase5_FMAMcount-1)
t_tabulado6 = stats.t.ppf(1 - alfa/2, fase6_FMAMcount-1)
t_tabulado7 = stats.t.ppf(1 - alfa/2, fase7_FMAMcount-1)
t_tabulado8 = stats.t.ppf(1 - alfa/2, fase8_FMAMcount-1)

testet_fase1 = fase1_FMAM / (fase1_FMAMstd / ((fase1_FMAMcount-1)**0.5))
testet_fase2 = fase2_FMAM / (fase2_FMAMstd / ((fase2_FMAMcount-1)**0.5))
testet_fase3 = fase3_FMAM / (fase3_FMAMstd / ((fase3_FMAMcount-1)**0.5))
testet_fase4 = fase4_FMAM / (fase4_FMAMstd / ((fase4_FMAMcount-1)**0.5))
testet_fase5 = fase5_FMAM / (fase5_FMAMstd / ((fase5_FMAMcount-1)**0.5))
testet_fase6 = fase6_FMAM / (fase6_FMAMstd / ((fase6_FMAMcount-1)**0.5))
testet_fase7 = fase7_FMAM / (fase7_FMAMstd / ((fase7_FMAMcount-1)**0.5))
testet_fase8 = fase8_FMAM / (fase8_FMAMstd / ((fase8_FMAMcount-1)**0.5))

#------------------------------------------------------------------------------#

# Figura histograma da quantidade de eventos.

fig, axs = plt.subplots(2, 1, dpi = 400)

bins = np.arange(0,9,1)
count_fase = [int(np.array(fase0_FMAMcount[0,0])), int(np.array(fase1_FMAMcount[0,0])), int(np.array(fase2_FMAMcount[0,0])),
              int(np.array(fase3_FMAMcount[0,0])), int(np.array(fase4_FMAMcount[0,0])), int(np.array(fase5_FMAMcount[0,0])), int(np.array(fase6_FMAMcount[0,0])),
              int(np.array(fase7_FMAMcount[0,0])), int(np.array(fase8_FMAMcount[0,0]))]

#print(np.array(count_fase).sum())

rects = axs[0].bar(bins, 
                   count_fase, 
                   color = 'gray', 
                   label = 'Percentual Mensal')

axs[0].bar_label(rects, padding=1.5, fontsize = 9)
axs[0].set_ylim(0,680)
axs[0].set_xticks(np.arange(0,9,1))
axs[0].set_xticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8'])
#axs[0].set_title("(a) Quantidade de dias absoluto para DJFMAM (1981 - 2022)", size=9, loc = 'left', weight = 'bold')
axs[0].set_title("(a) Absolute number of days for DJFMAM (1981 - 2022)", 
                 size=9, 
                 loc = 'left', 
                 weight = 'bold')
axs[0].set_ylabel("Quantidade (Dias)", size=10)
axs[0].set_xlabel("Fases", size=10)

# Distribuição relativa a quantidade total de dias com eventos

rects = axs[1].bar(bins, 
                   np.array((count_fase / np.array(count_fase).sum()) * 100).round(decimals=1), 
                   color = 'gray', 
                   label = 'Percentual Mensal')

axs[1].bar_label(rects, padding=1.5, fontsize = 9)
axs[1].set_ylim(0,30)
axs[1].set_xticks(np.arange(0,9,1))
axs[1].set_xticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8'])
#axs[1].set_title("(b) Proporção de dias com fases para DJFMAM (1981 - 2022)", size=9, loc = 'left', weight = 'bold')
axs[1].set_title("(b) Proportion of days with phases for DJFMAM (1981 - 2022)", size=9, loc = 'left', weight = 'bold')
axs[1].set_ylabel("%", size=10)
axs[1].set_xlabel("Fases", size=10)

plt.tight_layout()

plt.savefig('./figuras/histograma_DJFMAM_clim8122_precfunceme_ingles.png')

# Figur espacial dos trimestres

def cut2shapefile(plot_obj, shape_obj):
         
            """
            plot_obj: axis where plot is being made. ex: ax
            shape_obj: basemap shapefile. ex: m.nordeste_do_brasil when shape is read with m.readshapefile(path/to/nordeste_do_brasil, nordeste_do_brasil)
            """
         
            x0,x1 = plot_obj.get_xlim()
            y0,y1 = plot_obj.get_ylim()
             
            edges = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            edge_codes = [Path.MOVETO] + (len(edges) - 1) * [Path.LINETO]
             
            verts = shape_obj[0] + [shape_obj[0][0]]
            codes = [Path.MOVETO] + (len(verts) - 1) * [Path.LINETO]
             
            path = Path(verts+edges, codes+edge_codes)
         
            patch = PathPatch(path, facecolor='white', lw=0)
            plot_obj.add_patch(patch)
        
        # mascara shp
        
#shp para criar a mascara de interesse
shp = shpreader.Reader('/media/acer/FUNCEME/trabalhos/dataset/shp/EstadosBR_IBGE_LLWGS84.shp')
for f in shp.records():
        if f.attributes['ESTADO'] == 'CE':
          a = (f.geometry)
          x, y = a.exterior.coords.xy
          t = [[(a, s) for a, s in zip(x, y)]]

crs = ccrs.PlateCarree()

fname = r'/media/acer/FUNCEME/trabalhos/dataset/shp/regioes_hidrograficas.shp'

fig, ax= plt.subplots(3, 3, dpi=300, subplot_kw={'projection': crs})

lon = ds_prec['longitude']
lat = ds_prec['latitude']
lons, lats = np.meshgrid(lon, lat)

#ax.add_feature(shape_feature)

lev = np.arange(-2.5,2.505,0.005)

#c =['#3498ed','#4ba7ef', '#76bbf3','#93d3f6','#b0f0f7','#D6FFFF','#ffffff','#FFFFC5','#fbe78a', '#ff9d37', '#ff5f26', '#ff2e1b', '#ae000c']
bounds = np.arange(-2.0,2.005,0.005)
c = 'bwr_r'
# Neutro

ax[0,0].set_extent([-42., -36., -2, -8.])      
ax[0,0].add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)

campo = ax[0,0].contourf(lon,lat, fase0_FMAM, 
                         levels = bounds, transform=ccrs.PlateCarree(), 
                         cmap = c, extend = 'both')   

cut2shapefile(ax[0,0],  t)

shapefile = list(shpreader.Reader(fname).geometries())
ax[0,0].add_geometries(shapefile, ccrs.PlateCarree(), 
                       edgecolor='black', facecolor='none', 
                       linewidth=.50)

ax[0,0].set_title(f'Inativa', fontsize=8, weight= 'bold')
ax[0,0].set_xticks(np.arange(-42,-33, 3), crs=ccrs.PlateCarree())
ax[0,0].set_yticks(np.arange(-8.,0,2), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax[0,0].xaxis.set_major_formatter(lon_formatter)
ax[0,0].yaxis.set_major_formatter(lat_formatter)
#ax[0,0].set_xlabel('Longitude',fontsize=8)
ax[0,0].set_ylabel('Latitude',fontsize=8)

# Fase 1

ax[0,1].set_extent([-42., -36., -2, -8.])      
ax[0,1].add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)

campo = ax[0,1].contourf(lon,
                         lat, 
                         fase1_FMAM, 
                         levels = bounds, 
                         transform=ccrs.PlateCarree(), 
                         cmap = c, 
                         extend = 'both')   

mask1 = np.zeros_like(testet_fase1)
mask1[abs(testet_fase1) > t_tabulado1] = 1

mask1_lat = lats[mask1==1]
mask1_lon = lons[mask1==1]

ax[0,1].scatter(mask1_lon, 
                mask1_lat,  
                s=0.1,
                color = 'black',
                transform=ccrs.PlateCarree())

cut2shapefile(ax[0,1],  t)

shapefile = list(shpreader.Reader(fname).geometries())

ax[0,1].add_geometries(shapefile, ccrs.PlateCarree(), 
                       edgecolor='black', facecolor='none', 
                       linewidth=.50)

ax[0,1].set_title(f'Fase 1', fontsize=8, weight= 'bold')
ax[0,1].set_xticks(np.arange(-42,-33, 3), crs=ccrs.PlateCarree())
ax[0,1].set_yticks(np.arange(-8.,0,2), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax[0,1].xaxis.set_major_formatter(lon_formatter)
ax[0,1].yaxis.set_major_formatter(lat_formatter)
#ax[0,1].set_xlabel('Longitude',fontsize=8)
#ax[0,1].set_ylabel('Latitude',fontsize=8)

# Fase 2

ax[0,2].set_extent([-42., -36., -2, -8.])      
ax[0,2].add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)

campo = ax[0,2].contourf(lon,lat, fase2_FMAM, 
                         levels = bounds, transform=ccrs.PlateCarree(), 
                         cmap = c, extend = 'both')   

mask2 = np.zeros_like(testet_fase2)
mask2[abs(testet_fase2) > t_tabulado2] = 1

mask2_lat = lats[mask2==1]
mask2_lon = lons[mask2==1]

ax[0,2].scatter(mask2_lon, 
                        mask2_lat,  
                        s=0.1,
                        color = 'black',
                        transform=ccrs.PlateCarree())

cut2shapefile(ax[0,2],  t)

shapefile = list(shpreader.Reader(fname).geometries())
ax[0,2].add_geometries(shapefile, ccrs.PlateCarree(), 
                       edgecolor='black', facecolor='none', 
                       linewidth=.50)

ax[0,2].set_title(f'Fase 2', fontsize=8, weight= 'bold')
ax[0,2].set_xticks(np.arange(-42,-33, 3), crs=ccrs.PlateCarree())
ax[0,2].set_yticks(np.arange(-8.,0,2), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax[0,2].xaxis.set_major_formatter(lon_formatter)
ax[0,2].yaxis.set_major_formatter(lat_formatter)
#ax[0,2].set_xlabel('Longitude',fontsize=8)
#ax[0,2].set_ylabel('Latitude',fontsize=8)

# Fase 3

ax[1,0].set_extent([-42., -36., -2, -8.])      
ax[1,0].add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)

campo = ax[1,0].contourf(lon,lat, fase3_FMAM, 
                         levels = bounds, transform=ccrs.PlateCarree(), 
                         cmap = c, extend = 'both')

mask3 = np.zeros_like(testet_fase3)
mask3[abs(testet_fase3) > t_tabulado3] = 1

mask3_lat = lats[mask3==1]
mask3_lon = lons[mask3==1]

ax[1,0].scatter(mask3_lon, 
                        mask3_lat,  
                        s=0.1,
                        color = 'black',
                        transform=ccrs.PlateCarree())

cut2shapefile(ax[1,0],  t)

shapefile = list(shpreader.Reader(fname).geometries())
ax[1,0].add_geometries(shapefile, ccrs.PlateCarree(), 
                       edgecolor='black', facecolor='none', 
                       linewidth=.50)

ax[1,0].set_title(f'Fase 3', fontsize=8, weight= 'bold')
ax[1,0].set_xticks(np.arange(-42,-33, 3), crs=ccrs.PlateCarree())
ax[1,0].set_yticks(np.arange(-8.,0,2), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax[1,0].xaxis.set_major_formatter(lon_formatter)
ax[1,0].yaxis.set_major_formatter(lat_formatter)
#ax[1,0].set_xlabel('Longitude',fontsize=8)
ax[1,0].set_ylabel('Latitude',fontsize=8)

# Fase 4

ax[1,1].set_extent([-42., -36., -2, -8.])      
ax[1,1].add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)

campo = ax[1,1].contourf(lon,lat, fase4_FMAM, 
                         levels = bounds, transform=ccrs.PlateCarree(), 
                         cmap = c, extend = 'both')   

mask4 = np.zeros_like(testet_fase4)
mask4[abs(testet_fase4) > t_tabulado4] = 1

mask4_lat = lats[mask4==1]
mask4_lon = lons[mask4==1]

ax[1,1].scatter(mask4_lon, 
                        mask4_lat,  
                        s=0.1,
                        color = 'black',
                        transform=ccrs.PlateCarree())  

cut2shapefile(ax[1,1],  t)

shapefile = list(shpreader.Reader(fname).geometries())
ax[1,1].add_geometries(shapefile, ccrs.PlateCarree(), 
                       edgecolor='black', facecolor='none', 
                       linewidth=.50)

ax[1,1].set_title(f'Fase 4', fontsize=8, weight= 'bold')
ax[1,1].set_xticks(np.arange(-42,-33, 3), crs=ccrs.PlateCarree())
ax[1,1].set_yticks(np.arange(-8.,0,2), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax[1,1].xaxis.set_major_formatter(lon_formatter)
ax[1,1].yaxis.set_major_formatter(lat_formatter)
#ax[1,1].set_xlabel('Longitude',fontsize=8)
#ax[1,1].set_ylabel('Latitude',fontsize=8)

# Fase 5

ax[1,2].set_extent([-42., -36., -2, -8.])      
ax[1,2].add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)

campo = ax[1,2].contourf(lon,lat, fase5_FMAM, 
                         levels = bounds, transform=ccrs.PlateCarree(), 
                         cmap = c, extend = 'both')

mask5 = np.zeros_like(testet_fase5)
mask5[abs(testet_fase5) > t_tabulado5] = 1

mask5_lat = lats[mask5==1]
mask5_lon = lons[mask5==1]

ax[1,2].scatter(mask5_lon, 
                        mask5_lat,  
                        s=0.1,
                        color = 'black',
                        transform=ccrs.PlateCarree())  


cut2shapefile(ax[1,2],  t)

shapefile = list(shpreader.Reader(fname).geometries())
ax[1,2].add_geometries(shapefile, ccrs.PlateCarree(), 
                       edgecolor='black', facecolor='none', 
                       linewidth=.50)

ax[1,2].set_title(f'Fase 5', fontsize=8, weight= 'bold')
ax[1,2].set_xticks(np.arange(-42,-33, 3), crs=ccrs.PlateCarree())
ax[1,2].set_yticks(np.arange(-8.,0,2), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax[1,2].xaxis.set_major_formatter(lon_formatter)
ax[1,2].yaxis.set_major_formatter(lat_formatter)
#ax[1,2].set_xlabel('Longitude',fontsize=8)
#ax[1,2].set_ylabel('Latitude',fontsize=8)

# Fase 6

ax[2,0].set_extent([-42., -36., -2, -8.])      
ax[2,0].add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)

campo = ax[2,0].contourf(lon,lat, fase6_FMAM, 
                         levels = bounds, transform=ccrs.PlateCarree(), 
                         cmap = c, extend = 'both')

mask6 = np.zeros_like(testet_fase6)
mask6[abs(testet_fase6) > t_tabulado6] = 1

mask6_lat = lats[mask6==1]
mask6_lon = lons[mask6==1]

ax[2,0].scatter(mask6_lon, 
                        mask6_lat,  
                        s=0.1,
                        color = 'black',
                        transform=ccrs.PlateCarree())     

cut2shapefile(ax[2,0],  t)

shapefile = list(shpreader.Reader(fname).geometries())
ax[2,0].add_geometries(shapefile, ccrs.PlateCarree(), 
                       edgecolor='black', facecolor='none', 
                       linewidth=.50)

ax[2,0].set_title(f'Fase 6', fontsize=8, weight= 'bold')
ax[2,0].set_xticks(np.arange(-42,-33, 3), crs=ccrs.PlateCarree())
ax[2,0].set_yticks(np.arange(-8.,0,2), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax[2,0].xaxis.set_major_formatter(lon_formatter)
ax[2,0].yaxis.set_major_formatter(lat_formatter)
ax[2,0].set_xlabel('Longitude',fontsize=8)
ax[2,0].set_ylabel('Latitude',fontsize=8)

# Fase 7

ax[2,1].set_extent([-42., -36., -2, -8.])      
ax[2,1].add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)

campo = ax[2,1].contourf(lon,lat, fase7_FMAM, 
                         levels = bounds, transform=ccrs.PlateCarree(), 
                         cmap = c, extend = 'both') 

mask7 = np.zeros_like(testet_fase7)
mask7[abs(testet_fase7) > t_tabulado7] = 1

mask7_lat = lats[mask7==1]
mask7_lon = lons[mask7==1]

ax[2,1].scatter(mask7_lon, 
                        mask7_lat,  
                        s=0.1,
                        color = 'black',
                        transform=ccrs.PlateCarree())  

cut2shapefile(ax[2,1],  t)

shapefile = list(shpreader.Reader(fname).geometries())
ax[2,1].add_geometries(shapefile, ccrs.PlateCarree(), 
                       edgecolor='black', facecolor='none', 
                       linewidth=.50)

ax[2,1].set_title(f'Fase 7', fontsize=8, weight= 'bold')
ax[2,1].set_xticks(np.arange(-42,-33, 3), crs=ccrs.PlateCarree())
ax[2,1].set_yticks(np.arange(-8.,0,2), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax[2,1].xaxis.set_major_formatter(lon_formatter)
ax[2,1].yaxis.set_major_formatter(lat_formatter)
ax[2,1].set_xlabel('Longitude',fontsize=8)
#ax[2,1].set_ylabel('Latitude',fontsize=8)

# Fase 8

ax[2,2].set_extent([-42., -36., -2, -8.])      
ax[2,2].add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)

campo = ax[2,2].contourf(lon,lat, fase8_FMAM, 
                         levels = bounds, transform=ccrs.PlateCarree(), 
                         cmap = c, extend = 'both')


mask8 = np.zeros_like(testet_fase8)
mask8[abs(testet_fase8) > t_tabulado8] = 1

mask8_lat = lats[mask8==1]
mask8_lon = lons[mask8==1]

ax[2,2].scatter(mask8_lon, 
                        mask8_lat,  
                        s=0.1,
                        color = 'black',
                        transform=ccrs.PlateCarree())  

cut2shapefile(ax[2,2],  t)

shapefile = list(shpreader.Reader(fname).geometries())
ax[2,2].add_geometries(shapefile, ccrs.PlateCarree(), 
                       edgecolor='black', facecolor='none', 
                       linewidth=.50)

ax[2,2].set_title(f'Fase 8', fontsize=8, weight= 'bold')
ax[2,2].set_xticks(np.arange(-42,-33, 3), crs=ccrs.PlateCarree())
ax[2,2].set_yticks(np.arange(-8.,0,2), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax[2,2].xaxis.set_major_formatter(lon_formatter)
ax[2,2].yaxis.set_major_formatter(lat_formatter)
ax[2,2].set_xlabel('Longitude',fontsize=8)
#ax[2,2].set_ylabel('Latitude',fontsize=8)

cax = plt.axes([0.87, 0.1, 0.03, 0.8])

fig.colorbar(campo, cax=cax, ticks= np.arange(-2.0,2.02,0.2), label='mm/dia')

plt.subplots_adjust(top=0.95,
                    bottom=0.09,
                    left=0.025,
                    right=0.88,
                    hspace=0.4,
                    wspace=0.01)

plt.savefig('./figuras/composite_DJFMAM_clim8122_precfunceme_1%_shaded.png', dpi = 600)
