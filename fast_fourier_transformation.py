# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------------------------------------------------#
#       Script que aplica a Fast Fourier Transformation (FFT) nos dados de precipitação da Krigagem da FUNCEME           #
#                                                                                                                        #  
# Ref:                                                                                                                   #
#                                                                                                                        #
#     -- https://docs.scipy.org/doc/scipy/tutorial/fft.html                                                              #
#                                                                                                                        #
#  Script no final salva em .nc o arquivo filtrado e não filtrado  no período estabelecido no diretório dados da pasta   #
#                                                                                                                        #      
#------------------------------------------------------------------------------------------------------------------------#

# Créditos do script

__author__ = ["Francisco Agustinho de Brito Neto"]
__email__ = "francisco.brito@funceme.br"
__credits__ = ["Bruno Dias", "Letícia Karyne", "ChatGPT"]

# Bibliotecas 

import numpy as np
import xarray as xr
from scipy.fft import fft, ifft, fftfreq

# Caminho para o arquivo NetCDF com os dados de precipitação

nc_file_path = '/media/francisco.brito/FUNCEME/trabalhos/dataset/prec_funceme/' \
               'pr_daily_funceme_obs_19730101_20230410_kriging_valid_rain.nc'

# Abre o arquivo NetCDF como um conjunto de dados xarray

ds = xr.open_dataset(nc_file_path)

# Set variáveis

prec = ds['pr'].sel(time=slice('1981-01-01','2022-12-31'))

# Calculando anomalia usando a normal climatológica de 1981 - 2010

anom = prec.groupby('time.dayofyear') - prec.sel(time= slice('1981-01-01','2022-12-31')).groupby('time.dayofyear').mean('time')

# Seleciona a variável de precipitação e transforma em matriz numpy

precip0 = np.array(anom)

# Tratando os dados na

precip = np.nan_to_num(precip0, nan=0.0)

# Define a frequência de amostragem dos dados de precipitação
# Assumimos que os dados são diários, portanto a frequência de amostragem é 1/dia

dt = 1.0

# Calcula a transformada de Fourier dos dados de precipitação ao longo do eixo temporal

precip_fft = fft(precip, axis=0)

# Calcula as frequências correspondentes à transformada de Fourier

freqs = fftfreq(precip.shape[0], d=dt)

# Define os limites da faixa de frequência desejada (20 a 90 dias)

f_min = 1 / 90.0
f_max = 1 / 20.0

# Cria um filtro para selecionar apenas as frequências dentro da faixa desejada
filtro = np.logical_and(freqs >= f_min, freqs <= f_max)

# Aplica o filtro na transformada de Fourier dos dados de precipitação
precip_fft_filtrado = precip_fft.copy()
precip_fft_filtrado[np.logical_not(filtro), :, :] = 0

# Calcula a transformada inversa dos dados filtrados para obter o sinal filtrado
precip_filtrado = ifft(precip_fft_filtrado, axis=0)

# Cria uma nova matriz de dados xarray a partir dos dados filtrados

precip_filtrado_xr = xr.DataArray(precip_filtrado.real, dims=['time', 'latitude', 'longitude'],
                                  coords={'time': ds['time'].sel(time=slice('1981-01-01','2022-12-31')), 
                                          'latitude': ds['latitude'], 'longitude': ds['longitude']})

precip_nofiltrado_xr = xr.DataArray(precip0, dims=['time', 'latitude', 'longitude'],
                                  coords={'time': ds['time'].sel(time=slice('1981-01-01','2022-12-31')), 
                                          'latitude': ds['latitude'], 'longitude': ds['longitude']})

# Cria um novo conjunto de dados xarray com a matriz de dados filtrados
ds_filtrado = xr.Dataset({'precip': precip_filtrado_xr})
ds_nofiltrado = xr.Dataset({'precip': precip_nofiltrado_xr})

#print(ds_filtrado.min())

# Define o caminho para o arquivo NetCDF com os dados de precipitação filtrados
nc_file_path_filtrado = './dados/prec_FFTanom2090_scipy_CE_1981_2022_clim8122_daily_krigagem.nc'
nc_file_path_nofiltrado = './dados/prec_anom_CE_1981_2022_clim8122_daily_krigagem.nc'

# Salva o conjunto de dados filtrados em um novo arquivo NetCDF
ds_filtrado.to_netcdf(nc_file_path_filtrado)
ds_nofiltrado.to_netcdf(nc_file_path_nofiltrado)
