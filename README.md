# Smart-Map

## Descrição

Plugin para QGIS 3.10.x 64 bits que possibilita a predição e o mapeamento de atributos do solo. 
Permite a interpolação dos dados utilizando Krigagem Ordinária e técnicas de Machine Learning. Neste plugin foi implementado o Support Vector Machine (SVM). 
Possibilita a interação com o QGIS a partir de layers (Shapefile e GeoTIFF). 
O principal objetivo desse plugin é fornecer uma plataforma livre e open source, aliada às funções já existentes no QGIS. 
Disponível para os Sistemas Operacionais Windows, Linux e Mac tendo como única dependência o QGIS 3.10.x 64 bits instalado.
Desenvolvido em Python 3, o plugin utiliza alguns pacotes já incorporados ao seu código como: [scikit-learn](https://github.com/scikit-learn/scikit-learn), [scikit-fuzzy](https://github.com/scikit-fuzzy/scikit-fuzzy), [pysal](https://github.com/pysal) e [pyKrige](https://github.com/GeoStat-Framework/PyKrige).

## Como Instalar

### Dependências 

#### Windows

O Plugin tentará instalar a biblioteca scikit-learn automaticamente, mas caso ocorra alguma falha, 
abra o OSGeo4W Shell como administrador na pasta de atalhos do QGIS e digite os seguintes comandos:

`python -m pip install scikit-learn`

Se tiver algum erro do tipo: "PermissionError: [WinError 5] Access is denied", tente abrir o OsGeo Shell como administrador.

#### Linux e MacOS

No QGIS abra o Terminal Python e verifique se as bibliotecas python: scipy, pandas e sklearn estão instaladas.
Para verificar utilize os seguintes comandos:

`import pandas`

`import scipy`

`import sklearn`

Para cada linha de comando, digite o comando e tecle enter. Se ocorrer o erro: ModuleNotFoundError, ao tentar importar a biblioteca faça a instalação 
através do terminal Linux/MacOS ou pelo OsGeo Shell do QGIS. 

No terminal Linux ou MacOS utilize os seguintes comandos para instalar as bibliotecas python: 

Para instalar a biblioteca pandas: `python -m pip install pandas` ou `pip3 install pandas`

Para instalar a biblioteca scipy: `python -m pip install scipy` ou `pip3 install scipy`

Para instalar a biblioteca sklearn: `python -m pip install scikit-learn` ou `pip3 install scikit-learn`

Caso estes comandos não funcionem, verifique qual é a sua distribuição Linux ou MacOS e pesquise na internet como instalar bibliotecas python no Linux ou MacOS. 

### Instalação Ideal

#### Windows

Procure pelo nome Smart-Map no repositório oficial do Qgis (Plugins -> Intalar plugins) e clique em instalar.  

#### Linux e MacOs

Antes de procurar no repositório oficial do QGIS, certifique-se que as bibliotecas python: scipy, pandas e sklearn estão instaladas.
Siga as instruções descritas acima, em como instalar dependencias para Linux e MacOS. 
Após certificar que as bibliotecas estão instaladas, procure pelo nome Smart-Map no repositório oficial do Qgis (Plugins -> Intalar plugins) e clique em instalar.  

### Instalação Manual

1.	Baixe o arquivo Smart_Map.zip na página de [releases](https://github.com/gustavowillam/SmartMapPlugin/releases)
2.	Abra o Qgis 3, entre no menu Plugins->Manage and Install Plugins->Install From Zip File
3.	Na linha zip file, clique no botão "..." e escolha o arquivo zip baixado do plugin.

## Manual de Utilização 

Um tutorial simplificado de como usar esse complemento: [Wiki](https://github.com/gustavowillam/SmartMapPlugin/wiki)

## Reportar erros ou bugs

Caso você queira reportar algum erro não se esqueça de:

1.	Caso haja uma janela de erro python, copie todo o conteúdo dela
2.	Verifique no painel de logs do QGIS, que pode ser acessado no canto inferior direito do programa, as abas python error e Smart-map e copie todos os logs que existirem lá.
3.	Informe exatamente os passos que você fez e verifique se o erro continua acontecendo se você reiniciar o plugin, reinstalar, fechar abrir, etc...
4.	Diga o que aconteceu e o que você esperava acontecer.
5.	Se possível envie seu arquivo de projeto (zip e qgis) com as layers.

Para reportar algum erro use a página [issues](https://github.com/gustavowillam/SmartMapPlugin/issues). 


## QGIS

QGIS (anteriormente conhecido como Quantum GIS) é um software livre com código-fonte aberto, multiplataforma de sistema de informação geográfica (SIG) que permite a visualização, edição e análise de dados georreferenciados.

[Download](https://www.qgis.org/pt_BR/site/forusers/download.html#windows)- Qgis 3 for Windows 64 bits

[Download](https://qgis.org/en/site/forusers/alldownloads.html#debian-ubuntu)- Qgis 3 for Linux Debian/Ubuntu

[Download](https://qgis.org/en/site/forusers/download.html)- Other Downloads

## Links and Resources

[Qgis](https://www.qgis.org/) - A Free and Open Source Geographic Information System
[Python3](https://www.python.org/) - Python is a programming language that lets you work quickly and integrate systems more effectively

## Documentação:

[Python3](https://www.python.org/)

[Qt5 API](https://doc.qt.io/qt-5)

[Qgis 3.10.x python API](https://qgis.org/pyqgis/master/)

[PyQGIS developer cookbook](https://docs.qgis.org/3.10/en/docs/pyqgis_developer_cookbook/index.html)

## Licença  GPL 3

This program is free software; you can redistribute it and/or modify it under the terms of the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.pt-br.html) as published by the Free Software Foundation; 
either version 3 of the License, or (at your option) any later version.
