# coding: utf-8
"""
/***************************************************************************
#Arquivo: kriging.py

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
from scipy.spatial.distance import cdist
from . import variogram_models
import matplotlib.path as mplPath
from scipy.spatial import cKDTree
import scipy.linalg.lapack



class OrdinaryKriging:
    

    eps = 1.0e-10  # Cutoff for comparison to zero
    #dicionário com os modelos de semivariograma implementados
    variogram_dict = {
        "linear": variogram_models.linear_variogram_model,
        "linear-sill": variogram_models.linear_sill_variogram_model,
        "gaussian": variogram_models.gaussian_variogram_model,
        "spherical": variogram_models.spherical_variogram_model,
        "exponential": variogram_models.exponential_variogram_model,
        "hole-effect": variogram_models.hole_effect_variogram_model,
    }

    def __init__(
        self,
        xy,
        z,
        variogram_model,
        variogram_parameters):
        
        """
        Função de Inicialização da Classe OrdinaryKriging
        Parameters
        ----------
        xy é matriz n x 2 , com as coordenadas x e y dos pontos experimentais
        z é vetor com os valores de atributo para cada ponto experimetnal
        variogram_model é o modelo do semivariograma teorio
        variogram parameters é os parametros do modelo [nugget, range , sill]

        Returns
        -------
        None.

        """
        
        self.x=xy.iloc[:,0]
        self.y=xy.iloc[:,1]
        self.z=z.to_numpy()
        
        
        
        self.variogram_model_parameters=variogram_parameters
        
        # set up variogram model and parameters...
        self.variogram_model = variogram_model
        #configura a funcao semivariograma a ser executada
        self.variogram_function = self.variogram_dict[self.variogram_model]

        
   
    def Grid (self,pixel_x,pixel_y,has_contour,points):
        
        """
        Função para gerar a malha de pontos paara interpolação
        
        pixel_x é o tamanho da grade (pixel), em x
        
        pixel_y é o tamanho da grade (pixel), em y
        
        has_countour é uma variavel boleana, indicando se o contorno foi definido
        
       points é a matriz com os pontos de contorno (se foi definido), ou os pontos amostrais (se o
       contorno não foi definido)
       Primeira coluna é X e a segunda Y
        
        Return
        
        matriz n x 2 . Se o contorno foi definido, a matriz contêm apenas os pontos dentro do contorno.
        Se o contorno não foi definido, a matriz contem todos os pares de pontos no retangulo definido
        entre xmin e y min, até xmax e ymax
        """
 
        # Generate the grid to start the kriging process
        #Find min and max of contour
        x_min = points.iloc[:,0].min()
        x_max = points.iloc[:,0].max()
    
        y_min = points.iloc[:,1].min()
        y_max = points.iloc[:,1].max()
        
        #Generate the grid
        #+ grid it is to force use x_max. the stop is not used in arange
        gridx = np.arange(x_min, x_max, pixel_x)
        gridy = np.arange(y_min, y_max, pixel_y)
        
       
        # Generate a polygon with contour, if contour it is defined
        if has_contour:
            contours = mplPath.Path(np.array(points))
        
        # Generate a array with all grid points inside contour
        #gridxy it is a n x 2 , where n its the number o points
        gridxy=[]
        #
        # Run all combination of i and j points of grid
        for i in gridx:
            for j in gridy:
                # if point it is internal of contour and contour it is definided
                if has_contour:
                    if contours.contains_point((i,j)):
                        gridxy.append([i,j])
                
                #if has_countor is not defined
                else :
                    gridxy.append([i,j])
                    
        
        #return a nx2 array with pair of point internal of contour
        return np.array(gridxy)

    def _get_kriging_matrix(self, n):
        """
        Versão modificada da função do PyKrige
        Assembles the kriging matrix.
        
        Build the matrix C of kriging. Matrix with the covariance between 
        all experimental points, with n+1 x n+1 dimension
        
        """
        #forma vetor nx2
        xy = np.concatenate((self.x[:, np.newaxis], self.y[:, np.newaxis]), axis=1 )
        self.xy=xy
        #forma vetor de distancias nxn
        d = cdist(xy, xy, "euclidean")
        #
        a = np.zeros((n + 1, n + 1))
        nug,rang_,sill=self.variogram_model_parameters
        a[:n, :n] = -self.variogram_function(d,nug,rang_,sill)
        np.fill_diagonal(a, 0.0)
        a[n, :] = 1.0
        a[:, n] = 1.0
        a[n, n] = 0.0

        return a

 
    
    def _exec_loop(self,a_all,xypts,n):
            
        """
          
        Versão modificada da função do PyKrige
         
        Funcao que calcula o   valor inteprpolado e o desvio padrao da estimacao
        em cada ponto da grande
         
         
        a_alls é a matriz de covariancia entre todos pontos experimentais, com dimensão n+1
        
        xypts sao as coordenadas dos pontos da grade onde a interpolacao será realizada.
        
        n é o numero de pontos experimentais
        
        Retorna zvalue e sigma
        
        """
         
        #comprimento do vetor de pontos da grade
        npt= len(xypts)
        #Vetor para guardar resultados     
        zvalues=np.zeros(npt)
        sigmasq=np.zeros(npt)
        
        #monta arvore de procura de pontos experimentais
        #self.xy (obtido em _get_kriging_matrix), é vetor de pontos experimentais com
        #dimensão nx2 
        tree = cKDTree(self.xy)
        
        
        #p=2 use Euclian Distance
        #njobs=-1 use all processor of computer
        #distance_upper_bound, maxima distancia para achar vizinhos
        #query completa a matriz idx com o comprimento do vetor de pontos
        #experimentais, quanto não acha k vizinhos no raio definido
        dist_all,ids_all=tree.query(xypts,k=self.n_closest_points,p=2, 
                        distance_upper_bound=self.radius, n_jobs=-1)
        
        #calculada para 4 vizinhos, sem considerar radio de busca
        #Consume menos tempo de processamento
        dist_4n,ids_4n=tree.query(xypts,k=4,p=2,n_jobs=-1)
        
        #para cada ponto da grade
        for i in range(npt):
            
            #ids dos vizinhos, para o ponto i da grade 
            ids=ids_all[i]
            
            #distancia ate vizinhos, para o ponto i da grade 
            dist=dist_all[i]
            
            #remove pontos, que foram completados
            #query completa a matriz idx com o comprimento do vetor de pontos
            #experimentais, quanto não acha k vizinhos no raio definido
            idx_del = np.argwhere(ids == n)
            dist = np.delete(dist, idx_del)
            ids = np.delete(ids, idx_del)
             
             #numero vizinhos encontrados
            n_neig=len(ids)
            
               
             #Se achar menos que 4, procura os 4 mais proximos, sem considerar
             #o radio de busca
            if n_neig<4:
                 n_neig=4
                 #usa os valores já calculados previamente
                 ids=ids_4n[i]
                 dist=dist_4n[i]

   
             #Na matrix a (c), seleciona a covariancia dos pontos que são vizinhos
             #no ponto i
            a_selector = np.concatenate((ids, np.array([a_all.shape[0] - 1])))
            a = a_all[a_selector[:, None], a_selector]
             
            #valor de indice inferiores ao cutoof (eps) são adotados como zero
            if np.any(np.absolute(dist) <= self.eps):
                zero_value = True
                zero_index = np.where(np.absolute(dist) <= self.eps)
            else:
                 zero_index = None
                 zero_value = False
         
            b = np.zeros((n_neig + 1, 1))
            nug,rang_,sill=self.variogram_model_parameters
            b[:n_neig, 0] = -self.variogram_function(dist, nug,rang_,sill)
            if zero_value:
                b[zero_index[0], 0] = 0.0
            b[n_neig, 0] = 1.0
            
            x = scipy.linalg.solve(a, b)
            zvalues[i] = x[:n_neig, 0].dot(self.z[ids])
            sigmasq[i] = -x[:, 0].dot(b[:, 0])
   
         
        return zvalues, np.sqrt(sigmasq)

 
    def execute(
        self,
        xypoints,
        n_closest_points,
        radius):
        
        """
        Função que executa a interpolação por krigagem
        xpoints e ypoitns são as coordenadas x e y, dos pontos a serem interpolados (grade)
        
         
        n_closest_points é o número de vizinhos a serem usados na interpolação. 
        
        radius é o raio de busca por vizinhos
        
        ***A prioridade será o radio de busca. Se , para um raio de busca
        achar mais vizinhos que o configurado no  n_closest_points, usa o numero configurado.
        Caso contrário, usa o número que encontrar, com valor minimo de 4.
        
        Return
        
        zvalues vetor com os valores interpolado por krigagem ordinária 
        
        sigmasq desvio padrão assoiado com o valor interpolado
        
        """
        self.radius=radius
        self.n_closest_points=n_closest_points
        
       
        #comprimento do vetor de pontos experimentais
        n = len(self.x)
        
        #Get Matrix C for krigind (matrix of covariance between all experimental points)
        a = self._get_kriging_matrix(n)
      
        #chama funcao para executar krigagem em cada ponto do grid
        zvalues, sigmasq = self._exec_loop(a, xypoints,n)
 

        return zvalues, sigmasq