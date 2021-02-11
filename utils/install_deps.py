# -*- coding: utf-8 -*-
"""
/***************************************************************************
#Arquivo: install_deps.py

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
import sys
import platform
import subprocess
import zipfile

#import pathlib
#import pip
#from pip._internal import main
#import ctypes
#import time   
from qgis.PyQt.QtWidgets import QMessageBox
#from qgis.PyQt import QtCore


system = platform.system()  #[Windows, Linux, Darwin]
print('\nOperation System:', system)

#if system == 'Windows':
#    plugin_dir = pathlib.Path(__file__).parent
#    sys.path.append(plugin_dir)


requirements=["scipy", "pandas", "scikit-learn", "pyKrige", "pysal", "scikit-fuzzy", "scikit-optimize"]


#dica web para instalar external libs 
#https://gis.stackexchange.com/questions/196002/development-of-a-plugin-which-depends-on-an-external-python-library


#Windows 
#C:\Program Files\QGIS 3.10\apps\Python37\Lib\site-packages                                            #local de instalação das libs do plugin via OSGEO4W

#C:\Users\Gustavo\AppData\Roaming\Python\Python37\site-packages                                        #local de instalação das libs do plugin via arquivo bach     

#C:\Users\Gustavo\AppData\Roaming\QGIS\QGIS3\profiles\default\python\site-packages                     #local de instalação das libs do plugin via install_deps


#Linux
#/home/gustavo/.local/lib/python3.8/site-packages/                                                     #local de instalação das libs do plugin via terminal

def unzip_external_package(library_name, path_to_zip_file, directory_to_extract_to): 

    
    print('uncompressing:', library_name) 

    with zipfile.ZipFile(os.path.join(path_to_zip_file, library_name + '.zip'), 'r') as zip_ref:

        zip_ref.extractall(directory_to_extract_to)



def load_external_package(library_name, library_version, exec_number, unzip): 

    
    directory_to_extract_to = os.path.dirname(os.path.abspath(__file__))              #C:\Users\Gustavo\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\Smart_Map\utils
    if unzip == False: 
	    directory_to_extract_to = directory_to_extract_to[:-5]                        #C:\Users\Gustavo\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\Smart_Map\
    else: 
	    directory_to_extract_to = directory_to_extract_to[:-23]                       #C:\Users\Gustavo\AppData\Roaming\QGIS\QGIS3\profiles\default\python\
	
    directory_to_extract_to = os.path.join(directory_to_extract_to, 'site-packages')  #C:\Users\Gustavo\AppData\Roaming\QGIS\QGIS3\profiles\default\python\site-packages           
    print('directory_to_extract_to', directory_to_extract_to)

    
    #if first execute, unzip the folder sklearn
    if (int(exec_number) == 0) and (unzip == True):

        path_to_zip_file = os.path.dirname(os.path.abspath(__file__))                     #C:\Users\Gustavo\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\Smart_Map\utils
        path_to_zip_file = path_to_zip_file[:-5]                                          #C:\Users\Gustavo\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\Smart_Map\
        print('path_to_zip_file', path_to_zip_file)

        unzip_external_package(library_name, path_to_zip_file, directory_to_extract_to)


    directory_to_extract_to = directory_to_extract_to.replace('\\', '/')
    sys.path.append(directory_to_extract_to)     #sys.path.insert(0, directory_to_extract_to)	
    print(library_name + " loaded.       version: " + library_version  + " Path: " + directory_to_extract_to)




def install_external_package(library_name, library_version, exec_number, unzip): 
    
        
	if int(exec_number) == 0: 
	
		try:  
			
			#try to install sklearn 

			print('Installing ' + library_name)
			subprocess.check_call(["python", '-m', 'pip', 'install', '--user', library_version]) #install pkg 
			print(library_name + ' installed with sucess. version: ' + library_version)

		except:  


			#import numpy 
			#file = numpy.__file__                                                            #'C:\\PROGRA~1\\QGIS3~1.10\\apps\\Python37\\lib\\site-packages\\numpy\\__init__.py'
			#directory_to_extract_to = file[:-17]                                             #'C:\\PROGRA~1\\QGIS3~1.10\\apps\\Python37\\lib\\site-packages\\'
			#print('directory_to_extract_to', directory_to_extract_to)

			print(library_name + ' not installed.')

			#if not install, load the folder sklearn
			load_external_package(library_name, library_version, exec_number, unzip)



print("\nChecking dependencies!")


#scipy
try:
    import scipy
    print("scipy        already installed. version: " + scipy.__version__ + " Locale:" + scipy.__file__)    

#except ModuleNotFoundError: 
except Exception:

    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setText('scipy library not found. Please go to github: https://github.com/gustavowillam/SmartMapPlugin and see how to install python libraries.')
    msg_box.exec_()


#pandas
try:
    import pandas
    print("pandas       already installed. version: " + pandas.__version__ + " Locale:" + pandas.__file__)    

#except ModuleNotFoundError: 
except Exception:

    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setText('pandas library not found. Please go to github: https://github.com/gustavowillam/SmartMapPlugin and see how to install python libraries.')
    msg_box.exec_()



#sklearn 
try:
    import sklearn
    print("scikit-learn already installed. version: " + sklearn.__version__ + " Locale:" + sklearn.__file__)    

#except ModuleNotFoundError: 
except Exception:

    file_dir = os.path.dirname(os.path.abspath(__file__))                                   #get the directory of currenty file python in execute

    if os.path.isfile(os.path.join(file_dir, 'execute_number.txt')): 
        f = open(os.path.join(file_dir, 'execute_number.txt'), 'r')        
        exec_number = f.read()
        f.close()     
    else: 
        f = open(os.path.join(file_dir, 'execute_number.txt'), 'w')        
        exec_number = 0    
        f.write(str(exec_number))          
        f.close()             


    requirements=["scikit-learn==0.22.2"]
    dep = requirements[0]


    #install_external_package('sklearn', dep, exec_number, unzip=True)

    if system == 'Windows':                                                                #install if SO is Windows, else user must install sklearn via cmd.   
        load_external_package('sklearn', dep, exec_number, unzip=True)

    else:        

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText('sklearn library not found. Please go to github: https://github.com/gustavowillam/SmartMapPlugin and see how to install python libraries.')
        msg_box.exec_()
        
        
print("Dependencies checked!")


#else: #Linux or macOS
#main.main(['install', dep])
#pip.main( ['install', dep])               
#main(['install', dep])
#subprocess.check_call(["python", '-m', 'pip', 'install', '--user', library_version]) 
#pip.main(['install', library_version])
