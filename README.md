# ECG: detecção de picos R e variabilidade
Este repositório faz parte da entrega do projeto final da matéria de Processamento Digital de Sinais no segundo semestre de 2025.

# Como rodar o código?
-É necessário instalar as bibliotecas:
    - wfdb (que contém o banco de dados recoemndado para uso)
    - numpy (responsável por realizar operações matemáticas)
    - matplot ()
    - scipy (responsável pela filtragem, a qual será utilizada a butter (butterworth) e filtfilt)
    - neurokit2 e ts2vg (para detecção de picos R, análise de VFC e HRV, sendo a última uma biblioteca auxiliar para refinar o método de detecção dos picos)

- Para conseguir baixar as bibliotecas, é necessário realizar a instalação dos mesmos através do comando "pip install [ ]", por exemplo
    - pip install wfdb 
- É possível fazer todas as instalações direto, como
    - pip install wfdb numpy matplot scipy neurokit2 ts2vg