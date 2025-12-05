import wfdb                                 # Biblioteca especializada para ler dados do PhysioNet, base de dados que foi coletada as ECGs (MIT-BIH)
import numpy
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt   # bibliotecas de filtragem  
import neurokit2                            # Biblioteca para detecção de picos R e análise de VFC/HRV

#Objetivo
#identificar automaticamente os picos R (deflexões mais proeminentes do complexo QRS) 
#calcular os intervalos RR (tmepo entre batimentos)
#derivar informaçoes sobre ritmo cardiaco e variabilidade

#Processo
#aplicar tecnicas para localizar as posições de picos R (derivação, limiar adaptativo - Pan Tompkins)
#analise dos intervalos RR calcula frequencia cardíaca instantanea, BPM e variabilidade de ritmo cardiaco (HRV)

#Resultados
#sinal filtrado com sobreposição de picos R detectados
#tabelas com valores médios e desvio padrão dos intervalos RR
#gráficos temporais mostrando a variação de frequencia cardiaca ao longo do registro (histograma de distribuição dos intervalos RR e métricas de desempenho de detecção)

#Adicional
#análise espectral de HRV para ilustrar o equilíbrio autonômico e discutir implicações fisiológicas

# ------------------------ RECEBER AS INFORMAÇÕES DO BANCO DE DADOS ---------------------------------
taxa_amostragem = 360   # amostragem padrão do banco de dados
canal = 0               # banco de dados foi dividido em dois canais, um deles foi escolhido
registro_ID = '101'     # são 48 registros, escolhido o primeiro

record = wfdb.rdrecord(registro_ID, pn_dir= 'mitdb')
sinal_bruto = record.p_signal[:, canal]
fs = record.fs
'''
plt.figure(figsize=(12, 4))
plt.plot(numpy.arange(len(sinal_bruto)) / fs, sinal_bruto)
plt.title(f'Sinal ECG Bruto (Registro {registro_ID})')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude (mV)')
plt.show()
'''
# -------------------------------- PRÉ PROCESSAMENTO --------------------------------------------------
#filtragem passa-banda (0,5Hz a 40Hz) para eliminar ruídos de baixa frequencia e interferência de rede elétrica (50/60Hz)
freq_baixa = 0.5
freq_alta = 40
ordem_filtro = 4           # número de polos do filtro, inclinação com que o filtro atenua as frequencias fora da banda de passagem, (4, 5 ou 6) são as mais equilibradas para ECG

# Frequências de corte normalizadas (Nyquist = fs/2), no nosso caso, Nyquist = 180Hz
Wn = [freq_baixa / (fs / 2), freq_alta / (fs / 2)]

# Projetar o filtro Butterworth
# b = coefs do numerador
# a = coefs do denominador
# bandpass = passa banda
b, a = butter(ordem_filtro, Wn, btype='bandpass')

# Aplicar o filtro 
# filtfilt para aplicar o filtro na direção direta e reversa e, assim, cancelar o deslocamento de fase gerado pelo filtro digital
sinal_filtrado = filtfilt(b, a, sinal_bruto)
'''
plt.figure(figsize=(12, 4))
plt.plot(numpy.arange(len(sinal_bruto)) / fs, sinal_bruto, label='Bruto', alpha=0.5)
plt.plot(numpy.arange(len(sinal_filtrado)) / fs, sinal_filtrado, label='Filtrado')
plt.title('Comparação: Sinal Bruto vs. Sinal Filtrado (Passa-Banda)')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude (mV)')
plt.xlim(0, 5) # Zoom nos primeiros 5 segundos para melhor visualização
plt.legend()
plt.show()
'''
# ------------------------------------ DETECÇÃO DOS PICOS R -------------------------------------------
# método de detectar os picos R foi escolhido o padrão emrich2023 
# foi o mais preciso a partir dos dados inseridos e é mais recente que o Pan-Tompkins
# baseado no detector de Koka, transforma o ECG em representação gráfica e extrai as posições exatas usando grafos
_, info = neurokit2.ecg_peaks(sinal_filtrado, sampling_rate=fs, method='emrich2023')
picosR = info["ECG_R_Peaks"]

plt.figure(figsize=(12, 4))
plt.plot(numpy.arange(len(sinal_filtrado)) / fs, sinal_filtrado)
plt.plot(picosR / fs, sinal_filtrado[picosR], "o", color='red', markersize=5)
plt.title('Sinal ECG Filtrado com Picos R Detectados')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude (mV)')
plt.xlim(0, 20)
plt.show()
