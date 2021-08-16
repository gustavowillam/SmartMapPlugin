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


#comando para instalar libs via OSGEO4W  (Instalador .EXE)
#C:\Program Files\QGIS 3.10\apps\Python37\pip -m pip install <nome_da_lib>                             


#instalar sklearn no Python 3.9.5    (Instaldor .MSI) 
#ir na pasta: C:\Program Files\QGIS 3.16.9\apps\Python39>
#digitar:  pip install --pre -U scikit-learn



#check de library version
#<library>.__version__   


#check de library path                                                                              
#<library>.__file__ 																				   


###check python version in QGIS 

#from platform import python_version
#print(python_version())
#Saida: 3.7.0


#Table of Python and QGIS Versions 
#QGIS 3.10.12                     -> Python 3.7.0
#QGIS 3.16.9 (Instalador .EXE)    -> Python 3.7.0
#QGIS 3.16.9 (Instalador .MSI)    -> Python 3.9.5


###check QGIS Version  

#import qgis.utils
#qgis.utils.Qgis.QGIS_VERSION
#Saida: '3.16.3-Hannover'


#Windows 

#C:\Program Files\QGIS 3.10\apps\Python37\Lib\site-packages                                            #local de instalação das libs do plugin via OSGEO4W

#para instalar libs via OSGEO4W ir nas pastas:

#C:\Program Files\QGIS 3.10\apps\Python37\ 
#digitar: python -m pip install scikit-learn


#C:\Program Files\QGIS 3.16\apps\Python39\ 
#digitar: python -m pip install scikit-learn  (Instalador .EXE)
#digitar: pip install --pre -U scikit-learn   (Instalador .MSI)



#C:\Users\Gustavo\AppData\Roaming\Python\Python37\site-packages                                        #local de instalação das libs do plugin via arquivo bach     

#C:\Users\Gustavo\AppData\Roaming\QGIS\QGIS3\profiles\default\python\site-packages                     #local de instalação das libs do plugin via install_deps

#C:\Users\Gustavo\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins                           #local de instalação do plugin no QGIS 


#Linux
#/home/gustavo/.local/lib/python3.8/site-packages/                                                     #local de instalação das libs do plugin via terminal Linux

#MacOS
#/Applications/QGIS-LTR.app/Contents/MacOS/lib/python3.8/site-packages                                 #local de instalação das libs do plugin via terminal MacOS


def unzip_external_package(qgis_python_version, library_name, path_to_zip_file, directory_to_extract_to): 

    
    print('uncompressing:', library_name) 

    if "v3.9" in qgis_python_version:  	#Python 3.9 

        with zipfile.ZipFile(os.path.join(path_to_zip_file, library_name + '39.zip'), 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

    else:                               #Python 3.7 or 3.8 

        with zipfile.ZipFile(os.path.join(path_to_zip_file, library_name + '37.zip'), 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
	


def load_external_package(qgis_python_version, library_name, library_version, exec_number, unzip): 

    
    directory_to_extract_to = os.path.dirname(os.path.abspath(__file__))              #C:\Users\Gustavo\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\Smart_Map\utils
    if unzip == False: 
	    directory_to_extract_to = directory_to_extract_to[:-5]                        #C:\Users\Gustavo\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\Smart_Map\
    else: 
	    directory_to_extract_to = directory_to_extract_to[:-23]                       #C:\Users\Gustavo\AppData\Roaming\QGIS\QGIS3\profiles\default\python\
	
    directory_to_extract_to = os.path.join(directory_to_extract_to, 'site-packages')  #C:\Users\Gustavo\AppData\Roaming\QGIS\QGIS3\profiles\default\python\site-packages           
    print('directory_to_extract_to', directory_to_extract_to)

    
    #if first execute, unzip the folder sklearn
    if (int(exec_number) == 0) and (unzip == True):

        path_to_zip_file = os.path.dirname(os.path.abspath(__file__))                 #C:\Users\Gustavo\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\Smart_Map\utils
        path_to_zip_file = path_to_zip_file[:-5]                                      #C:\Users\Gustavo\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\Smart_Map\
        print('path_to_zip_file', path_to_zip_file)

        unzip_external_package(qgis_python_version, library_name, path_to_zip_file, directory_to_extract_to)


    #directory_to_extract_to = directory_to_extract_to.replace('\\', '/')             #não funciona no python 3.9 
    sys.path.append(directory_to_extract_to)     
    print(library_name + " loaded.       version: " + library_version  + " Locale: " + directory_to_extract_to)




def install_external_package(library_name, library_version, exec_number, unzip): 
    
        
	if int(exec_number) == 0: 
	
		try:  
			
			#try to install sklearn 

			print('Installing ' + library_name)
			subprocess.check_call(["python", '-m', 'pip', 'install', '--user', library_version]) #install pkg 
			print(library_name + ' installed with sucess. version: ' + library_version)

		except:  


			#import numpy 
			#file = numpy.__file__                                                         #'C:\\PROGRA~1\\QGIS3~1.10\\apps\\Python37\\lib\\site-packages\\numpy\\__init__.py'
			#directory_to_extract_to = file[:-17]                                          #'C:\\PROGRA~1\\QGIS3~1.10\\apps\\Python37\\lib\\site-packages\\'
			#print('directory_to_extract_to', directory_to_extract_to)

			print(library_name + ' not installed.')

			#if not install, load the folder sklearn
			load_external_package(qgis_python_version, library_name, library_version, exec_number, unzip)


###check QGIS (32 or 64 bits) and Python Version
#import sys
#sys.version
#Saida: '3.7.0 (v3.7.0:1bf9cc5093, Jun 27 2018, 04:59:51) [MSC v.1914 64 bit (AMD64)]'     #instalador .exe
#Saida: '3.7.0 (v3.7.0:1bf9cc5093, Jun 27 2018, 04:06:47) [MSC v.1914 32 bit (Intel)]'     #instalador .exe
#Saida: '3.9.5 (tags/v3.9.5:0a7dcbd, May  3 2021, 17:27:52) [MSC v.1928 64 bit (AMD64)]'   #instalador .msi

			

#check qgis architecture 
qgis_architecture   = sys.version  
#check Python Version
qgis_python_version = sys.version 


if "32 bit" in qgis_architecture: 

    print("\nQGIS 32 bit")	

    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setText('This plugin only works on QGIS 64-bit. Please install 64-bit QGIS before installing the plugin.')
    msg_box.exec_()

else: 	
	
	print("\nQGIS 64 bit")	
		
	print("\nChecking dependencies!")


	#scipy
	try:
		import scipy
		print("scipy        already installed. version: " + scipy.__version__ + "  Locale:" + scipy.__file__)    

	#except ModuleNotFoundError: 
	except Exception:

		msg_box = QMessageBox()
		msg_box.setIcon(QMessageBox.Warning)
		msg_box.setText('scipy library not found. Please go to github: https://github.com/gustavowillam/SmartMapPlugin and see how to install python libraries.')
		msg_box.exec_()


	#pandas
	try:
		import pandas
		print("pandas       already installed. version: " + pandas.__version__ + "  Locale:" + pandas.__file__)    

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

		file_dir = os.path.dirname(os.path.abspath(__file__))                               #get the directory of currenty file python in execute

		if os.path.isfile(os.path.join(file_dir, 'execute_number.txt')): 
			f = open(os.path.join(file_dir, 'execute_number.txt'), 'r')        
			exec_number = f.read()
			f.close()     
		else: 
			f = open(os.path.join(file_dir, 'execute_number.txt'), 'w')        
			exec_number = 0    
			f.write(str(exec_number))          
			f.close()             

		if "v3.9" in qgis_python_version:
			requirements=["scikit-learn==0.24.2"]
		else: 
			requirements=["scikit-learn==0.22.2"]

		dep = requirements[0]


		#install_external_package('sklearn', dep, exec_number, unzip=True)

		if system == 'Windows':                                                             #install if SO is Windows, else user must install sklearn via cmd.   
			load_external_package(qgis_python_version, 'sklearn', dep, exec_number, unzip=True)

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
