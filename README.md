# Smart-Map

## Descrição

Plugin para QGIS 3.10.x que possibilita a predição e o mapeamento de atributos do solo. 
Permite a interpolação dos dados utilizando Krigagem Ordinária e técnicas de Machine Learning. Neste plugin foi implementado o Support Vector Machine (SVM). 
Possibilita a interação com o QGIS a partir de layers (Shapefile e GeoTIFF). 
O principal objetivo desse plugin é fornecer uma plataforma livre e open source, aliada às funções já existentes no QGIS. 
Disponível para os Sistemas Operacionais Windows, Mac e Linux tendo como única dependência o QGIS instalado.
Desenvolvido em Python 3, o plugin utiliza alguns pacotes já incorporados ao seu código como: scikit-learn, scikit-fuzzy, pysal e pyKrige.

## Como Instalar

### Dependências 

O Plugin tentará instalar a biblioteca scikit-learn automaticamente, mas caso ocorra alguma falha, 
abra o OSGeo4W Shell como administrador na pasta de atalhos do QGIS e digite os seguintes comandos:

python -m pip install scikit-learn

Se tiver algum erro do tipo: "PermissionError: [WinError 5] Access is denied", tente abrir o OsGeo Shell come administrador.
Para que a instalação automática funcione, basta executar o Qgis como administrador e instalar o plugin.

### Instalação Ideal

Procure pelo nome Smart-Map no repositório oficial do Qgis (Plugins -> Intalar plugins) e clique em instalar.  

### Instalação Manual

1.	Baixe o arquivo Smart_Map.zip na página de [releases](https://github.com/gustavowillam/SmartMapPlugin/releases)  ou nesse [link](https://github.com/gustavowillam/SmartMapPlugin/releases/download/v1.0/Smart_Map.zip)
2.	Abra o Qgis 3, entre no menu Plugins->Manage and Install Plugins->Install From Zip File
3.	Na linha zip file, clique no botão "..." e escolha o arquivo zip baixado do plugin.

### Instalação "Forçada"

Coloque estes arquivos dentro da pasta Smart_Map ou crie essa pasta, no caminho:
Linux: ~/.qgis/plugins/python/
Windows: C:\Users\USER\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins
onde USER é seu nome de usuário

## Manual de Utilização 

Um tutorial simplificado de como usar esse complemento: [Wiki](https://github.com/gustavowillam/SmartMapPlugin/wiki)

## Reportar erros ou bugs

Caso você queira reportar algum erro não se esqueça de:

1.	Caso haja uma janela de erro python, copie todo o conteúdo dela
2.	Verifique no painel de logs do QGIS, que pode ser acessado no canto inferior direito do programa, as abas python error e Smart-map e copie todos os logs que existirem lá.
3.	Informe exatamente os passos que você fez e verifique se o erro continua acontecendo se você reiniciar o plugin, reinstalar, fechar abrir, etc...
4.	Diga o que aconteceu e o que você esperava acontecer.
5.	Se possível envie seu arquivo de projeto (zip e qgis) com as layers.

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

[Qgis 3.10.x python API](https://qgis.org/pyqgis/master/)

[PyQGIS developer cookbook](https://docs.qgis.org/3.10/en/docs/pyqgis_developer_cookbook/index.html)

## Licença  GPL 3

This program is free software; you can redistribute it and/or modify it under the terms of the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.pt-br.html) as published by the Free Software Foundation; 
either version 3 of the License, or (at your option) any later version.
