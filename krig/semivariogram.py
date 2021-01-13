# -*- coding: utf-8 -*-
"""
/***************************************************************************
#Arquivo: semivariogram.py

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

import os 
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import itertools as it
from scipy import spatial
from . import variogram_models


file_dir = os.path.dirname(os.path.abspath(__file__))                          #get the directory of currenty file python in execute

name_log = 'log.txt'

f = open(os.path.join(file_dir, name_log), 'w')                                #open(name_log, "w")
f.write('\nArquivo de Log SmartMap\n')
f.close()

def Log(msg):
    
    f = open(os.path.join(file_dir, name_log), "a")
    f.write(msg+'\n')
    f.close()

class Semivariogram:
    
    #dicionário com os modelos de semivariograma implementados
    variogram_dict = {
        "linear": variogram_models.linear_variogram_model,
        "linear-sill": variogram_models.linear_sill_variogram_model,
        "gaussian": variogram_models.gaussian_variogram_model,
        "spherical": variogram_models.spherical_variogram_model,
        "exponential": variogram_models.exponential_variogram_model,
        "hole-effect": variogram_models.hole_effect_variogram_model,
    }
    
    def __init__(self,xy,z):
        """
        Função de Inicialização da Classe Semivariogram
        Parameters
        ----------
        xy é matriz n x 2 , com as coordenadas x e y dos pontos experimentais
        z é vetor com os valores de atributo para cada ponto experimetnal

        Returns
        -------
        None.

        """
        
        Log('\nIniciando Geração do Semivariograma\n')
        
        
        #Constroe um banco de dados no pandas
        self.var=pd.DataFrame().astype('float32')
        #Columa do banco de dados com as distâncias
        self.var['lag']=spatial.distance.pdist(xy, metric='euclidean')
        #Columa com todas as semivariâncias
        self.var['gamma']=[(y - x)**2 for x, y in it.combinations(z, 2)]
        
        #Mímina e máxima distância entre pontos experimentais
        self.min_dist=self.var['lag'].min()
        #Usado para calcular a distância ativa
        self.max_dist=self.var['lag'].max()
        #Calcula a Variancia da Amostra
        self.sample_variance=z.var()        
 
    
    def Exp_Semiv(self, dist_lag,act_dist): # 
        
        """
        Função para gerar o semivariograma experimental, com base nos ponto xy e no valor de atributo z
        dist_lag é o comprimento do lag definido pelo usuário
        
        Retorna o vetor de distância média, a semivariância e o número de pontos usados para construir
        o semivarigorama
        """
        var=self.var
        
        #Ordena em ordem crescente de acordo com a distância
        #O vetor gamam acompanha a ordenação
        var=var.sort_values(by='lag')
        #Remove os pontos com distância > distância ativa
        remove_index=var[var['lag'] > act_dist ].index
        var=var.drop(remove_index)
        
        #Vetor para classificar as distância
        #Mudança :Agora distância inicia na Mimima Distancia. Antes era Zero
        #bins=np.arange(0,max(var['lag']), dist_lag)
        bins=np.arange(self.min_dist,max(var['lag']), dist_lag)

        #Classifica as distâncias conforme o vetor
        ind = np.digitize(var['lag'],bins)
        
        #Agrupa o vetor de distâncias , obtendo-se o valor médio
        lag=var['lag'].groupby(ind).mean()
        self.lag=lag.to_numpy() #convert pandas series to numpy
        #group gamma e calculate mean()/2, acoording matheron estimator
        #agrupa as semivariâncias, obtendo o indice de matheron
        
        gamma=var['gamma'].groupby(ind).mean().div(2)
        self.gamma=gamma.to_numpy() #convert pandas series to numpy
        
        #obtem o número de pares de pontos usado para construir cada lag
        npoints=var['lag'].groupby(ind).count()
        
        #retorna ao usuario
        return self.lag.astype('float32'),self.gamma.astype('float32'),npoints
        
    
    
    def Gamma(self,model,parameter):
        
        """
        A partir do vetor de distâncias (self.lag),
        calcula o vetor de semivariância teorica, a partir da distância
        model é o modelo de semivariograma escolhido
        [linear, linear-sill,spherical,gaussian,exponential,hole-effect]
        parameter é o efeito pepita, alcance e patamar do modelo escolhido
        
        Retorno
        gamma T é a semivariância teórica
        rss é a soma de quadrado do resíduo
        r2 é o coeficiente de determinação
        
        """
        
        func=self.variogram_dict[model]
        
        gammaT=func(self.lag,parameter[0],parameter[1],parameter[2])
        
        rss=np.sum((self.gamma-gammaT)**2)
        tss=np.sum((self.gamma-np.mean(self.gamma))**2)
        
 
        return gammaT, rss, 1-(rss/tss)
        
        
    def Fit(self,list_model):
        
        """
        Função para ajustar o modelo teório com base no lag e gamma 
        
        list_model é a lista de modelo a ser usada no ajuste
        [linear, linear-sill,spherical,gaussian,exponential,hole-effect]
        
        Return
        
        Dicionário com o resultado do ajuste para a lista de modelos
        
        a chave de cada elemento o nome do modelo
        
        o valor de cada elemento é uma lista com 
        [efeito pepita, alcance, patamar, soma quadrado residuo , r2]
        
       
        """
        #Adjust the theoretical semivariogram model
        lag=self.lag
        gamma=self.gamma
        nlag=len(lag)
        
        
        #lag[1]=np.inf
        #gamma[2]=np.nan
        
        #Pick a random initial value for the nugget
        Nugget=(gamma[1]*lag[0]-gamma[0]*lag[1])/(lag[0]-lag[1])
        if Nugget<0: Nugget=gamma[0]
        #Pick a random initial value for the sill
        Sill=(gamma[nlag-3]+gamma[nlag-2]+gamma[nlag-1])/3.0             
        #kick the starting value for the range
        Range=lag[int(nlag/2)]
        
        #Array of Initial Values. Also used in Gold Rule Fit                                                            
        self.init_vals = [Nugget, Range, Sill]
        
        #define the maximum values
        maxlim=[max(gamma),max(lag),max(gamma)]
        
        #dictyonary for results
        dict_results={}
        
        #
        for model in list_model:

            Log('\n\nPara o modelo:'+ model+'\n')

            check=True #option of curve_fit to check finite values

            func=self.variogram_dict[model]
            
            #First using Curve Fit and Check_Finite=True
            try:
                #
                Log('Usando Curve Fit and Check_Finite:'+ str(check))
                
                Log('\nChutes Iniciais : Nugget: '+str(Nugget)+'  Sill: '+str(Sill)+ '  Range: '+str(Range))

                Log('\nlag , Gamma')
                
                for i in range(len(lag)): 
                    
                    Log (str(lag[i]) +',' +str(gamma[i]))

                #return Nugget, Range , Sill and estimated covariance (not used)
                [Nugget,Range,Sill], _ = curve_fit(func, lag, gamma,method='trf', check_finite = check, p0=self.init_vals ,bounds=(0, maxlim) )
         

            except Exception:

                
                '''
                Log('ValueError at Curve Fit:  ydata or xdata contain NaNs, or if incompatible options are used. Change Check for false')
                
                Log('\nChutes Iniciais : Nugget: '+str(Nugget)+'  Sill: '+str(Sill)+ '  Range: '+str(Range))

                Log('\nlag , Gamma')
                
                for i in range(len(lag)): Log (str(lag[i]) +',' +str(gamma[i]))

                try :
                    
                     check=False #option of curve_fit to check finite values
                     Log('Usando Curve Fit and Check_Finite:'+ str(check))
                    
                     #return Nugget, Range , Sill and estimated covariance (not used)
                     [Nugget,Range,Sill], _ = curve_fit(func, lag, gamma,method='trf', check_finite = check, p0=self.init_vals ,bounds=(0, maxlim) )

                except ValueError:
                    Log('ValueError at Curve Fit :  ydata or xdata contain NaNs, or if incompatible options are used. Change to Golden Rule')
                    
                    
                    Nugget,Range,Sill=self.Gold_Rule(model)
                '''    

                    
            #except RuntimeError:

                Log('Error at Curve Fit: least-squares minimization fails. Change to Golden Rule')
                 
                Log('Chutes Iniciais : Nugget: '+str(Nugget)+' Sill: '+str(Sill)+ ' Range: '+str(Range))
                 
                Log ('\nlag , Gamma')
                
                for i in range(len(lag)): 
                    
                    Log (str(lag[i]) +',' +str(gamma[i]))

                Nugget,Range,Sill = self.Gold_Rule(model)
              
          
            else:

                Log('Nenhum erro na Curve Fit ')
                
            finally:

                #Calculate residual sum of square , whre error = gamma experimental - gamma fit
                _,rss,r2=self.Gamma(model,[Nugget,Range,Sill])
                dict_results [model]  = [Nugget,Range,Sill,rss,r2]
        
        Log('\nAjuste Finalizado\n')
        return dict_results
    
    
    def gold(self,ivar,xlow, xhigh, model, x,  y, z, maxIt, es):
        
        fp = -1;
         
        #Inicialization
        r = 0.618033989
        xl = xlow
        xu = xhigh
        iiter = 1
        d = r*(xu-xl)
        x1 = xl+d
        x2 = xu-d
          
        if 'x' in ivar:
            _,f1,_=self.Gamma(model,[x1,y,z])
            _,f2,_=self.Gamma(model,[x2,y,z])

           
        elif 'y' in ivar:
            _,f1,_=self.Gamma(model,[x,x1,z])
            _,f2,_=self.Gamma(model,[x,x2,z])
        
        elif 'z' in ivar:
            _,f1,_=self.Gamma(model,[x,y,x1])
            _,f2,_=self.Gamma(model,[x,y,x2])
            
            
            
        if (f1*fp > f2*fp) : xopt = x1
        else : xopt = x2
        
        
        while True:
            d = r*d
            xint = xu-xl
            
            if (f1*fp > f2*fp) :
                xl = x2
                x2 = x1
                x1 = xl+d
                f2 = f1
                
                if 'x' in ivar:
                     _,f1,_=self.Gamma(model,[x1,y,z])
                     
                elif 'y' in ivar:
                     _,f1,_=self.Gamma(model,[x,x1,z])
                     
                elif 'z' in ivar:
                     _,f1,_=self.Gamma(model,[x,y,x1])

            else:
                xu = x1
                x1 = x2
                x2 = xu-d
                f1 = f2
                
                if 'x' in ivar:
                     _,f2,_=self.Gamma(model,[x2,y,z])
                     
                elif 'y' in ivar:
                     _,f2,_=self.Gamma(model,[x,x2,z])
                     
                elif 'z' in ivar:
                     _,f2,_=self.Gamma(model,[x,y,x2])
                     
            iiter=iiter+1
            if (f1*fp > f2*fp) : xopt = x1
            else : xopt = x2
            
            # Check for stop
            if (xopt != 0.0) : ea = (1-r)*abs(xint/xopt)*100
            if (ea <= es or iiter >= maxIt) : break
        
        return xopt            

    def Gold_Rule(self,model):
        
       
        maxIt=25
        es=0.01
        imaxit = 25   # maxumum interation of gold rule
        j = 1;
       
        lag=self.lag
        gamma=self.gamma
        nlag=len(lag)
      
        #Pick a random initial value for the nugget
        Nugget=self.init_vals[0]
        Range=self.init_vals[1]
        Sill=self.init_vals[2]

       
        #Calculate residual sum of square , whre error = gamma experimental - gamma fit
        _,fant,_=self.Gamma(model,[Nugget,Range,Sill])
        
        while True:
            ivar = 'x'
            xL=0.00001
            xU= (gamma[nlag-3]+gamma[nlag-2]+gamma[nlag-1])/3.0
            Nugget=self.gold(ivar, xL, xU, model, Nugget, Range, Sill, maxIt, es)
            
            #
            ivar = 'y'
            xL=0.00001
            xU=lag[nlag-1]
            Range=self.gold(ivar, xL, xU, model, Nugget, Range, Sill, maxIt, es)
            
               #
            ivar = 'z'
            xL=(gamma[0]+gamma[1])/2
            xU=1.5*(gamma[nlag-4]+gamma[nlag-3]+gamma[nlag-2]+gamma[nlag-1])/4.0
            Sill=self.gold(ivar, xL, xU, model, Nugget, Range, Sill, maxIt, es)
            
            
            if (fant !=0):
                j=j+1
                _,fxyz,_=self.Gamma(model,[Nugget,Range,Sill])
                error = 100 * abs((fant - fxyz) / fant);
                fant = fxyz;
                if ((j >= imaxit) or (error < es)) : break
                    
                   
        Log("Ajuste com sucesso Usando a Gold Rule")
        return Nugget,Range,Sill
    

