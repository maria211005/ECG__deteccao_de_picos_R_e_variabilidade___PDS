# ECG: detecção de picos R e variabilidade
Este repositório faz parte da entrega do projeto final da matéria de Processamento Digital de Sinais no segundo semestre de 2025.

# Resumo do projeto
Trata-se de um algoritmo capaz de identificar picos R - deflexões mais proeminentes do complexo QRS, principais deflexões do eletrocardiograma - a partir de um banco de dados de registros de eletrocardiogramas, o qual foi solicitado para filtrar os sinais com o filtro passa-banda, encontrar tais picos e analisar os intervalos de um ponto R a outro, descrito como intervalo RR, podendo assim encontrar a variabilidade do ritmo cardíaco e da frequência cardíaca

# Processo
- filtragem passa-banda (0,5Hz a 40Hz) para eliminar ruídos de baixa frequencia e interferência de rede elétrica (50/60Hz)
- aplicar tecnicas para localizar as posições de picos R (nesse caso foi usada a Emrich 2023)
- com a analise dos intervalos RR é possível calcular frequencia cardíaca instantânea, BPM e variabilidade de ritmo cardiaco (HRV)

# Resultados
- gráfico do sinal inicial recebido do banco de dados
- sinal filtrado sem e com sobreposição de picos R detectados
- tabelas com valores médios e desvio padrão dos intervalos RR
- gráficos temporais mostrando a variação de frequencia cardiaca ao longo do registro 
- histograma de distribuição dos intervalos RR 
- métricas de desempenho de detecção

# Como rodar o código?
- É necessário instalar as bibliotecas:
    - *wfdb*: que contém o banco de dados recoemndado para uso
    - *numpy*: responsável por realizar operações matemáticas
    - *matplot*: usado para plotar os gráficos utilizados
    - *scipy*: responsável pela filtragem, a qual será utilizada a butter (butterworth) e filtfilt
    - *neurokit2* e *ts2vg*: para detecção de picos R, análise de VFC e HRV, sendo a última uma biblioteca auxiliar para refinar o método de detecção dos picos

- Para conseguir baixar as bibliotecas, é necessário realizar a instalação dos mesmos através do comando "*pip install [ ]*", por exemplo
    - *pip install wfdb*
- É possível fazer todas as instalações direto, como
    - *pip install wfdb numpy matplot scipy neurokit2 ts2vg*