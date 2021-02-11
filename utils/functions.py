# -*- coding: utf-8 -*-
"""
/***************************************************************************
#Arquivo: functions.py

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
import pandas as pd 


#Import the model to use Support Vector Machine
from sklearn import svm
from sklearn.model_selection import GridSearchCV

#from sklearn.metrics import mean_absolute_error  #EAM
from sklearn.metrics import mean_squared_error    #RMSE
#from sklearn.metrics import r2_score             #R2

#para o calculo do R2 da regressão da validação cruzada  (ajustar R2 negativos -> R2 positivos)
from sklearn.linear_model import LinearRegression

#from scipy import stats                          #EM
#from scipy.stats import kurtosis                 #Curtose

#para o calculo do índice de moran (correlação espacial entre as variáveis)

#from pysal.lib.weights import Kernel  (import all library )
from ..pysal.lib.weights.distance import Kernel         

#from pysal.explore import esda  (import all library )
from ..pysal.explore.esda.moran import Moran, Moran_BV   

'''
#para otimização dos parametros da SVM
from skopt import gp_minimize

#validação cruzada para seleção de modelo -> método optmize 
from sklearn.model_selection import cross_val_score 
'''

#função para calculo do IDW 
def IDW_Gustavo(Z, b):
    """
    Inverse distance weighted interpolation.
    Input
      Z: a list of lists where each element list contains 
         four values: X, Y, Value, and Distance to target
          point. Z can also be a NumPy 2-D array.
      b: power of distance
    Output
      Estimated value at the target location.
    """
    zw = 0.0                # sum of weighted z
    sw = 0.0                # sum of weights
    N = len(Z)              # number of points in the data
    for i in range(N):
        d = Z[i][3]
        if d == 0:
            return Z[i][2]
        w = 1.0/d**b
        sw += w
        zw += w*Z[i][2]
    return zw/sw


def idw(dist, values, weight_IDW):

    dist_pow = np.power(dist, weight_IDW)
    nominator = np.sum(values/dist_pow)
    denominator = np.sum(1/dist_pow)
    if denominator > 0:
        return nominator/denominator
    else:
        return 0 #none


def mean(dist, values, weight_IDW):

    #return sum(values)/len(values)
    return np.mean(values)


def svr_param_selection(norm, features, labels, kfolds):

    Cs     = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1, 10]

    param_grid = {'C': Cs, 'gamma' : gammas}
    
    '''
    for cont in (range(len(features))):

        train_features = np.copy(features)                            #copia o train_features para train_features2
        train_labels = np.copy(labels)                                #copia o train_labels para train_labels2
        
        
        train_features = np.delete(train_features, (cont), axis=0)         #deleta a linha cont da matriz - train_features         
        train_labels = np.delete(train_labels, (cont), axis=0)             #deleta a linha cont da matriz - train_features         
    
              
        norm = norm.fit(train_features)                  #normalizar o vetor de features 
        train_features = norm.transform(train_features)  #normalizar o vetor de features 
        

        grid_search = GridSearchCV(svm.SVR(kernel='rbf'), param_grid, cv=kfolds)
        grid_search.fit(train_features, train_labels)
        best_params = grid_search.best_params_
    
    
        if cont == 0:  #inicia a matriz de covariaveis p 
            list_C     = np.copy(best_params['C']) 
            list_gamma = np.copy(best_params['gamma'])  
        else:       
            list_C      = np.vstack((list_C,    best_params['C']))  #concatena após ultima linha.
            list_gamma  = np.vstack((list_gamma,best_params['gamma']))  #concatena após ultima linha.   
        
    list_C     = np.array(list_C)
    list_gamma = np.array(list_gamma) 
    '''
     
    train_features = np.copy(features)               #copia o train_features para train_features2
    train_labels = np.copy(labels)                   #copia o train_labels para train_labels2
    
           
    norm = norm.fit(train_features)                  #normalizar o vetor de features 
    train_features = norm.transform(train_features)  #normalizar o vetor de features 
    

    grid_search = GridSearchCV(svm.SVR(kernel='rbf'), param_grid, cv=kfolds)
    grid_search.fit(train_features, train_labels)
    best_params = grid_search.best_params_


    list_C     = np.copy(best_params['C']) 
    list_gamma = np.copy(best_params['gamma'])  
   
    
    return list_C.mean(), list_gamma.mean()  


#Import the model to use Random Forest 
from sklearn.ensemble import RandomForestRegressor


def param_selection_RF(features, labels, nfolds):

    
    N_estimators  = [10, 100, 500, 1000]   #nr. de arvores na floresta 
    Max_depths    = [3 , 5  , 10 , 15  ]   #profundidade das arvores 

    param_grid = {'n_estimators': N_estimators, 'max_depth' : Max_depths}
   
 
    train_features = np.copy(features)                            #copia o train_features para train_features2
    train_labels = np.copy(labels)                                #copia o train_labels para train_labels2
    
    
    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=nfolds)
    grid_search.fit(train_features, train_labels)
    best_params = grid_search.best_params_


    list_n_estimators     = np.copy(best_params['n_estimators']) 
    list_max_depth = np.copy(best_params['max_depth'])  
        

    list_n_estimators     = np.array(list_n_estimators)
    list_max_depth = np.array(list_max_depth) 
  
    
    return list_n_estimators.mean(), list_max_depth.mean()  


'''     
def svr_param_selection_optimize(norm, features, labels, kfolds): 


    def treinar_modelo_optimize(params): 
    
           
        model = svm.SVR(kernel = 'rbf', C = params[0],  gamma = params[1])
          
        scores = cross_val_score(estimator = model, X = train_features, y = train_labels, scoring = 'r2', cv = kfolds)
        mean_score = np.mean(scores)
    
        return -mean_score

    
    train_features = np.copy(features)                            #copia o train_features para train_features
    train_labels = np.copy(labels)                                #copia o train_labels para train_labels
             
    norm = norm.fit(train_features)                               #normalizar o vetor de features 
    train_features = norm.transform(train_features)               #normalizar o vetor de features 


    #para otimização dos parâmetros do SVM utilizando scipy-scikit-optimize
    params = [ (0.001, 10),  #C 
               (0.001, 10)]  #gamma 
    
    results = gp_minimize(func = treinar_modelo_optimize, dimensions = params, random_state=1, verbose = 1, n_calls = 30, n_random_starts=10)  
   
    return results.x[0], results.x[1] 
'''  
def concordance_correlation_coefficient(y_true, y_pred, sample_weight=None,  multioutput='uniform_average'):
    """Concordance correlation coefficient.
    The concordance correlation coefficient is a measure of inter-rater agreement.
    It measures the deviation of the relationship between predicted and true values
    from the 45 degree angle.
    Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.  
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.
    Examples
    --------
    >>> from sklearn.metrics import concordance_correlation_coefficient
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> concordance_correlation_coefficient(y_true, y_pred)
    0.97678916827853024
    """
   
    cor=np.corrcoef(y_true,y_pred)[0][1]
    
    mean_true=np.mean(y_true)
    mean_pred=np.mean(y_pred)
    
    var_true=np.var(y_true)
    var_pred=np.var(y_pred)
    
    sd_true=np.std(y_true)
    sd_pred=np.std(y_pred)
    
    numerator=2*cor*sd_true*sd_pred
    
    denominator=var_true+var_pred+(mean_true-mean_pred)**2

    return abs(numerator/denominator)
    


def calculate_statistics(labels_CV, labels):

      
    mean_labels = labels.mean()
            
    SQTot = 0
    SQRes = 0
    
    EM  = 0
    EAM = 0
    #RMSE = 0 

    num_tot = 0   #calculo de R2 Elp 
    den_tot = 0   #calculo de R2 Elp

    for j in range(len(labels)):

        num = (labels_CV[j] - mean_labels)**2
        den = (labels[j] - mean_labels)**2
        num_tot = num_tot + num 
        den_tot  = den_tot + den        
        
        SQt = (labels[j]-mean_labels)**2
        SQTot = SQTot + SQt 

        SQr = (labels[j]-labels_CV[j])**2
        SQRes = SQRes + SQr 
        
        EM = EM + (labels[j]-labels_CV[j])
        
        EAM = EAM + abs(labels[j]-labels_CV[j])
        

    #EM_calc = (labels_CV.std())/ np.sqrt(len(labels))          #EM_calc <> EM_lib 
    #EM_lib = stats.sem(labels_CV)    
   
    
    #EAM_calc = EAM / len(labels)                               #EAM_calc = EAM_lib 
    #EAM_lib = mean_absolute_error(labels, labels_CV)
  
    
    #RMSE_calc = np.sqrt(SQRes/len(labels))                     #RMSE_calc = RMSE_lib 
    RMSE_lib = np.sqrt(mean_squared_error(labels_CV, labels))    
   
    

    #Curtose = kurtosis(labels_CV) 
  
    
    #R2_calc = (1-(SQRes/SQTot))                                #R2_calc = R2_lib 
    #R2_lib = r2_score(labels, labels_CV)
  
    
           
    #convertendo o array 1d em array 2d, para realizar regressão linear 
    n = len(labels)

    labels_CV_2d =  labels_CV 
    labels_CV_2d.reshape(n,1)

    labels_2d = labels
    labels_2d.reshape(n,1)


    #para realizar regressão sobre os dados 
    regressor = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
    
    #ajustando o modelo  
    regressor.fit(labels_CV_2d.reshape(-1,1).astype(np.float32), labels_2d.reshape(-1,1))   
    
    #To retrieve the intercept:
    #intercept = regressor.intercept_
    
    #For retrieving the slope:
    #regression_coef = regressor.coef_
    
    #To retrieve the residual da regressão:
    #residual = regressor._residues


    #To retrieve the R2 da regressão:
    R2_RCV = regressor.score(labels_CV_2d.reshape(-1,1), labels_2d.reshape(-1,1))  #mede o score (R2) da validação cruzada 

    if den_tot > 0: 
        R2_Elp = num_tot / den_tot  
    else: 
        R2_Elp = 0
  
    lccc = 0 #concordance_correlation_coefficient(labels.reshape(-1), labels_CV.reshape(-1))  #Lins concordance        
    
    #return R2_lib, EM_lib, EAM_lib, RMSE_lib, Curtose,  R2_RCV
    return RMSE_lib, R2_RCV, regressor, R2_Elp, lccc 


###############################################################################

def calculate_index_moran(df, coord_x, coord_y, v_target): 

    y = np.array(df[v_target])  #valores de v_target 
    #list(y)
    
    points = np.column_stack([df[coord_x], df[coord_y]])   #coordenadas x, y 
      
    kw = Kernel(points)   #matriz de pesos entre das coordenadas x, y
    #list(kw)  
   
    np.random.seed(10)
    
    #mi = esda.Moran(y, kw, permutations = 999)    #Moran Index (when import all library)
    #mi = pysal.Moran(y, kw) 
    mi = Moran(y, kw, permutations = 999)    #Moran Index 
    
    #print(mi.I, mi.p_sim)
   
    return mi.I, mi.p_sim       


###############################################################################

def calculate_index_moran_BV(df_orig, coord_x, coord_y, v_target): 


    df = df_orig.copy() 
   
    tot_nan = df.isnull().sum().sum()
    if tot_nan > 0: 
        
        df = df.fillna(df.mean())   #preeenche com o valor da média da feature 


    y = np.array(df[v_target])  #valores de v_target 
    #list(y)
    
    points = np.column_stack([df[coord_x], df[coord_y]])   #coordenadas x, y 
      
    kw = Kernel(points)   #matriz de pesos entre das coordenadas x, y
    #list(kw)  
   
    np.random.seed(10)
    
    lista = [] 

    for c in range(len(df.columns)): 
        
        col_name  = df.columns[c]           #obtem o nome da coluna cujo indice é col    

        #if ((col_name != coord_x) and (col_name != coord_y)): 

        x = np.array(df[col_name])    #para calculo de Moran Bivariado
    
        #moran_global_BV = esda.Moran_BV(y, x, kw, transformation = 'r', permutations = 999)    #Moran Index (when import all library)
        moran_global_BV = Moran_BV(y, x, kw, transformation = 'r', permutations = 999)    #Moran Index 

   
        lista.append([col_name, moran_global_BV.I, moran_global_BV.p_sim])


    lista = np.array(lista)

    df_moran = pd.DataFrame(np.atleast_2d(lista), columns=['Covariavel', 'Moran', 'p-value'])
      
    #Um erro é gerado quando, na tabela de atributos, existe campo em que o indice de Moran não pode ser calculado                                                
    #df_moran["Moran"] = pd.to_numeric(df_moran["Moran"])
    df_moran["Moran"] = pd.to_numeric(df_moran["Moran"], errors='ignore', downcast='float')    
    #df_moran["p-value"] = pd.to_numeric(df_moran["p-value"])
    df_moran["p-value"] = pd.to_numeric(df_moran["p-value"], errors='ignore', downcast='float')    
    
    df_moran.sort_values(['Moran'], ascending=[False], inplace=True) 
  
    return df_moran       


###############################################################################

def eliminar_outlier(df, colname):
    
    if (colname != 'ID') and (colname != 'x') and (colname != 'y'):
        #z_min  = df[v_target].min()
        #z_max  = df[v_target].max()
        z_mean = df[colname].mean()  #media
        z_std  = df[colname].std()   #desvio padrao
        #z_var  = df[v_target].var()   #coeficiente de variação
        
        z_corte_min = z_mean - 2.5 * z_std
        z_corte_max = z_mean + 2.5 * z_std 
           
        lista_outlier =  []
        for i in range(len(df)):
            if ((df.iloc[i][colname] < z_corte_min) or (df.iloc[i][colname] > z_corte_max)): 
                 lista_outlier.append(i)
                 
        df = df.drop(df.index[lista_outlier])

    return df



###############################################################################
#Selected Features - Recursive Feature Elimination 
 
#gerar o modelo para aplicar o sklearn.feature_selection.RFE
from sklearn.feature_selection import RFE #, RFECV
from sklearn.preprocessing import StandardScaler           #para normalizar os dados de diferentes atributos para gerar ML com SVM   
 

def selected_features_RFE(df, coord_x, coord_y, v_target): 

    cols = list(df.columns)

    if 'ID' in cols: 
        cols.remove('ID')
        
    if 'ID_SM' in cols: 
        cols.remove('ID_SM')

    if coord_x in cols: 
        cols.remove(coord_x)

    if coord_y in cols: 
        cols.remove(coord_y)

    if v_target in cols: 
        cols.remove(v_target)
    
    df = df[[v_target] + cols]
    
    df = eliminar_outlier(df, v_target)

    X = df[cols]
    y = df[[v_target]]

    norm = StandardScaler()              
    
    C_average, gamma_average = svr_param_selection(norm, X, np.array(y).ravel(), 5)
    
    #print('Average Params SVR: C:' + str(C_average) + ' gamma: ' + str(gamma_average))       
    
    estimator = svm.SVR(kernel = 'linear', C = C_average, gamma = gamma_average)

    
    #norm = StandardScaler()              
    #norm = norm.fit(X)                               #normalizar o vetor de features 
    #X_Scaled = norm.transform(X)                     #normalizar o vetor de features 
    
    #RFE  with CV 
    #selector  = RFECV(estimator, step=1, cv=5, scoring='r2',     verbose=1)
    
    #RFE 
    selector = RFE(estimator,   step=1, n_features_to_select=5, verbose = 1)    

    
    selector = selector.fit(X, np.array(y).ravel())
    
    
    #selector.support_
    
    #selector.n_features_
    
    #selector.ranking_
    
    #selector.grid_scores_
    
    #selector.estimator_
    
    selected_features = list(X.iloc[:, selector.support_].columns)
    

    #selected_features_df = np.column_stack([cols, selector.ranking_])
    #selected_features_df = pd.DataFrame(np.atleast_2d(selected_features_df), columns=['Feature', 'Ranking'])   
    #selected_features_df["Ranking"] = pd.to_numeric(selected_features_df["Ranking"])
    #selected_features_df.sort_values(['Ranking'], ascending=[True], inplace=True) 


    #score = selector.score(X, y)
    
    #print("Optimal number of features : %d" % selector.n_features_)
    #print("Score of features selected : %f" % score)

    
    # Plot number of features VS. cross-validation scores
    #plt.figure()
    #plt.xlabel("Number of features selected")
    #plt.ylabel("Cross validation score (nb of correct classifications)")
    #plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
    #plt.show()


    #Features Importance using Random Forest 
    X  = df[selected_features]
    y  = df[[v_target]]
    n_estimators_average, max_depth_average = param_selection_RF(X, np.array(y).ravel(), 5)  
    
    #print('Average Params RF: n_estimators: ' + str(n_estimators_average) + ' max_depth: ' + str(max_depth_average))   
    
    X_train = X
    y_train = y 
    
    #Instantiate model with 1000 decision trees
    estimator = RandomForestRegressor(n_estimators = int(n_estimators_average), max_depth = int(max_depth_average)) 
    
    #ajustar o classificador aos dados de treinamento
    estimator.fit(X_train, y_train)
    
    #Get numerical feature importances  
    importances = list(estimator.feature_importances_)
    
    
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(selected_features, importances)]
    
    
    #Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    #feature_importances[0][1]
    
    # Print out the feature and importances 
    #[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

    
    df_RFE = pd.DataFrame(feature_importances, columns = ['Covariável' , 'Importância'])  

    return df_RFE        
        
        
        
###############################################################################
#calculo de FPI NCE  para determinar o número de clusters do FuzzyCmeans 
    

from scipy.spatial import distance
from math import log


# função para determinação da função de minimização 
def J(m,X,U,MI):
    # busca o número de objetos (linhas de dados) e o número de classes
    N,K=np.shape(U)
    # inicializa a variável que calculará a função de minimizaçãao
    myJ=0
    # iniciando o loop para cada objeto
    for i in range(N):
        # iniciando o lop para cada classe
        for j in range(K):
            # calcula da distância o objeto i até a classe j
            dij=distance.euclidean(X[i],MI[j])
            # atualiza o valor da função de minimização
            myJ=myJ+dij*dij*U[i][j]**m  
    # retorna o valor da função de minimização
    return myJ
#
# função para determinação da matriz de pertinência
def MP(m,X,MI):
    #busca o número de objetos (linhas de dados)
    N,L=np.shape(X)
    # busca o número de classes
    K,L=np.shape(MI)
    # inicializa a matriz de pertinência
    U = np.zeros(shape=(N,K))
    # iniciando o loop para cada objeto
    for i in range(N):
        # iniciando o loop para cada classe
        for j in range(K):
            # determinando a distância dij do objeto i até o centroide j 
            dij=distance.euclidean(X[i],MI[j])
            # testa o valor da distância
            if(dij!=0.0):
                # se a distância dij for diferente de zero,
                # inicia o procedimento de cálculo zerando a 
                # variável sum e inicia o loop para os
                # diferentes centroides
                sum=0
                for l in range(K):
                    # calcula a distância dil entre o objeto i e o centroide l
                    dil=distance.euclidean(X[i],MI[l])
                    # se a distância for zero, sai do loop
                    if(dil==0.0): break
                    # se é diferente de zero atualiza sum
                    sum=sum+(dij/dil)**(2/(m-1))
                # se a distância dil for diferente de zero, calcula o
                # valor do elemento ij da matriz de pertinência como sendo 1/sum
                if(dil!=0.0):
                    U[i][j]=1/sum
                # se dil é igual a zero, o elemento ij da matriz de pertinência é zero.
                else:
                    U[i][j]=0.0
            # se a distância dij for zero, é porque o valor na matriz de 
            # pertinência é igual a um
            else:
                U[i][j]=1.0
    # retorna a matriz de pertinência
    return U
#
# função para atualizar o centroide dos agrupamentos
def CENT(m,X,U):
    # determina o número de objetos (N), o número de variáveis (J)
    # e o número de classes (K)
    N,J=np.shape(X)
    L,K=np.shape(U)
    # inicializa a posição dos centroides com zeros
    MI=np.zeros(shape=(K,J))
    # inicia o loop dos centroides
    for j in range(K):
        # zera a função que calcula o somatório para o denominador
        sum=0
        # inicia o loop dos objetos
        for i in range(N):
            # atualiza o valor do numerado da fração que define o centroide
            MI[j]=MI[j]+(U[i][j]**m)*X[i]
            # atualiza o somatório para o denominador
            sum=sum+U[i][j]**m
        # calcula o valor do centroide
        MI[j]=MI[j]/sum
    # retorna a matriz com os centroides
    return MI
#
# função que realiza a análise de agrupamentos pelo método fuzzy k-means
def fuzzKMeans(m,K,n_int,tol,X):
    # definie o número de objetos e o número de feições (variáveis)
    N,NF=np.shape(X)
    # incializa a posição do centroides com zeros
    MI= np.zeros(shape=(K,NF))
    # sorteia K pontos para servir como centroide inicialmente
    random_index = np.random.choice(N,K,replace=False)
    # atualiza a posição dos centroides com os pontos sorteados
    for j in range(K):
        MI[j]=X[random_index[j]]
    # inicializa a variável contadora
    icount=0
    # inicializa o valor da função a ser minimizada com um valor muito grande
    JAnt=1.0E99
    # inicializa a lista que conterá os valores calculados da função de minimização
    Jplot=[]
    # inicia o loop de minimização da função
    while True:
        # atualiza a matriz de pertinência
        U = MP(m,X,MI)
        # atualiza a posição dos centroides
        MI = CENT(m,X,U)
        # atualiza o valor da função de minimização
        JAtual=J(m,X,U,MI)
        # adiciona o valor da função calculada à lista Jplot
        Jplot.append(JAtual)
        # verifica se já foi atingido o número máximo de iterações,
        # se sim, sai do loop
        if(icount>=n_int): break
        # verifica de a variação do valor da função de minimização
        # é menor que a tolerância, se sim sai do loop
        if(abs(JAnt-JAtual)<tol): break
        # atualiza o contador do loop
        icount=icount+1
        # atualiza o valor anterior da função de minimização
        JAnt=JAtual
    # cria uma lista para definir a classe de cada objeto (linha de dados)
    classe=[]
    # inicia o loop de cada objeto
    for i in range(N):
        # verifica qual foi o local em que a matriz de pertinência foi máxima
        # para cada objeto
        classe.append(U[i].argmax())
    # inicializa os somatórios que serão usados para cálculo do
    # fator de performance nebuloso (FPI) e da 
    # entropia normalizada de classificação (NCE) 
    sum1=0
    sum2=0
    # inicia o loop do objeto
    for i in range(N):
        # incia o loop das classes
        for j in range(K):
            # atualiza os somatórios
            sum1=sum1+U[i][j]*U[i][j] 
            sum2=sum2+U[i][j]*log(U[i][j])
    # calcula o valor do FPI
    FPI=1-(K*((1/N)*sum1)-1)/(K-1)
    # calcula o valor de NCE
    NCE=(-1/N)*sum2/log(K)
    # retorna FPI, NCE, a matriz de pertinência, a matriz de centroides,
    # a lista dos valores da função de minimização, a lista das classes
    # designadas ara cada objeto
    return FPI,NCE,U,MI,Jplot,classe

