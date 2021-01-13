# coding: utf-8
"""
/***************************************************************************
#Arquivo: variogram_models.py

                                 A QGIS plugin

                              -------------------
        begin                : 2018-08-15
        git sha              : $Format:%H$
        copyright            : (C) 2018 by Gustavo Willam Pereira
                                           Domingos Sárvio Magalhães Valente 
                                           Daniel Marçal de Queiroz
                                           Andre Luiz de Freitas Coelho
                                           Sandro Manuel Carmelino Hurtado
        email                : gustavowillam@gmail.com
 ***************************************************************************/
"""


import numpy as np

def linear_variogram_model(d, nugget,range_,psill):
    """ Linear model   """
    

    slope = (psill - nugget ) / range_
    return slope * d + nugget 

def linear_sill_variogram_model(d, nugget,range_,psill):
    """Linear with sill model"""

    slope = (psill - nugget ) / range_
    return np.where(d <=range_,slope * d + nugget,psill)


def hole_effect_variogram_model(d, nugget,range_,psill):
    """Hole Effect model]"""

    return np.full(len(d),(psill+nugget)/2)



# Semivariogram function
# Function for spherical semivariogram modeling
#d it a numpy array with lags
#Nugget, Range and Sill its is the parameter of model
#return it is the semivariance calculate using spherical model
def spherical_variogram_model(d,nugget,range_,psill):
    
    #if h < range, use spherical equation, else (h>=range) adopted sill
    return np.where(d<range_,nugget+(psill-nugget)*(1.5*d/range_-0.5*d**3/(range_**3)),psill)


# Function for exponential semivariogram modeling
#d it a numpy array with lags
#Nugget, Range and Sill its is the parameter of model
#return it is the semivariance calculate using exponential model
def exponential_variogram_model(d,nugget,range_,psill):
    
    return  nugget+(psill-nugget)*(1-np.exp(-3.0*d/range_))


# Function for gauss semivariogram modeling
#d it a numpy array with lags
#Nugget, Range and Sill its is the parameter of model
def gaussian_variogram_model(d,nugget,range_,psill):
    
    
    return nugget+(psill-nugget)*(1-np.exp(-3.0*d**2/(range_**2)))


