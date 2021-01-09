# tcc-facial-expression-recognition

NO meu TCC, eu desenvolvi uma rede neural convolucional (do zero) conectada a uma MLP para realizar o conhecimento de expressões faciais com hardware limitado (sem uso de GPU e frameworks).

O código é dividido em dois datasets:

1. [JAFFE](https://doi.org/10.5281/zenodo.3451524)
2. [IMPA-FACE3D](http://app.visgraf.impa.br/database/faces)

Os arquivos finalizados em *cnn* estão relacionados a rede convolucional (Convolutional Neural Networks) e os terminados em *fn* são as redes totalmente conectadas (Fully connected Networks).

Este repositório é baseado em Python usando os seguintes pacotes:

- [Matplotlib](https://matplotlib.org/)
- [Numpy](https://numpy.org/)
- [OS](https://docs.python.org/3/library/os.html)
- [Pandas](https://pandas.pydata.org/)

*OBS. 1: Os códigos salvos neste repositório consideram somente o Processo 2 (deslocamento em todas as direções), caso queira alterar isso leia os comentários das classes JAFFE() e IMPA() (começo dos arquivos *cnn*).*


*OBS. 2: Devido a direitos de imagem, os dois datasets não estão disponíveis no repositório, é necessário requerê-los e baixá-los.*
<!-- 
You can also compile and minify it for production
```
yarn build
``` -->

Entre em contato comigo por [Linkedin](https://www.linkedin.com/in/fabiobends/) ou mande um e-mail para fabiobends@gmail.com :man_technologist: 

Made with :heart:
