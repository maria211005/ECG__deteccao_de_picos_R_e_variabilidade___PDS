import wfdb                                 
from wfdb.processing import compare_annotations
import numpy
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt   
import neurokit2                            
import ts2vg                                

# ------------------------ RECEBER AS INFORMAÇÕES DO BANCO DE DADOS ---------------------------------
canal = 0               
registro_ID = '100'

record = wfdb.rdrecord(registro_ID, pn_dir= 'mitdb')
sinal_bruto = record.p_signal[:, canal]
fs = record.fs

plt.figure(figsize=(12, 4))
plt.plot(numpy.arange(len(sinal_bruto)) / fs, sinal_bruto)
plt.title(f'Sinal ECG Bruto (Registro {registro_ID})')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude (mV)')
plt.xlim(0, 20)
plt.savefig(f"{registro_ID} - plot_inicial.png")
plt.close()

# -------------------------------- PRÉ PROCESSAMENTO --------------------------------------------------
freq_baixa = 0.5
freq_alta = 40
ordem_filtro = 4           

Wn = [freq_baixa / (fs / 2), freq_alta / (fs / 2)]
b, a = butter(ordem_filtro, Wn, btype='bandpass') 
sinal_filtrado = filtfilt(b, a, sinal_bruto)

plt.figure(figsize=(12, 4))
plt.plot(numpy.arange(len(sinal_filtrado)) / fs, sinal_filtrado, label='Filtrado')
plt.title('Sinal Filtrado (Passa-Banda)')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude (mV)')
plt.xlim(0, 20)
plt.savefig(f"{registro_ID} - sinal_filtrado.png")
plt.close()

# ------------------------------------ DETECÇÃO DOS PICOS R -------------------------------------------
# método de detectar os picos R foi escolhido o padrão emrich2023 
# foi o mais preciso a partir dos dados inseridos e é mais recente que o Pan-Tompkins
# baseado no detector de Koka, transforma o ECG em representação gráfica e extrai as posições exatas usando grafos
_, info = neurokit2.ecg_peaks(sinal_filtrado, sampling_rate=fs, method= 'emrich2023')
picosR = info["ECG_R_Peaks"]

plt.figure(figsize=(12, 4))
plt.plot(numpy.arange(len(sinal_filtrado)) / fs, sinal_filtrado)
plt.plot(picosR / fs, sinal_filtrado[picosR], "o", color='red', markersize=5)
plt.title('Sinal ECG Filtrado com Picos R Detectados')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude (mV)')
plt.xlim(0, 20)
plt.savefig(f"{registro_ID} - picos_R.png")
plt.close()

#----------------------- VARIABILIDADE DA FREQUENCIA CARDIACA ----------------------------
diferencas = numpy.diff(picosR)
intervalosRR_s = diferencas/fs
intervalosRR_ms = intervalosRR_s * 1000

FC_instantanea = 60/intervalosRR_s

mediaRR = numpy.mean(intervalosRR_ms)
desvioRR = numpy.std(intervalosRR_ms)
mediaBPM = numpy.mean(FC_instantanea)

print("\n================== Tabela 1 =====================")
print("||          Resumo dos Intervalos RR           ||")
print("||                                             ||")
print(f"|| Frequência Cardíaca Média        {mediaBPM:6.2f} BPM ||")
print(f"|| Média dos Intervalos RR          {mediaRR:6.2f} ms  ||")
print(f"|| Desvio Padrão dos Intervalos RR  {desvioRR:6.2f} ms  ||")
print("=================================================")

# ----------------- CONSTRUÇÃO DOS GRÁFICOS TEMPORAIS E HISTOGRAMA -----------------------
tempo_picos = picosR / fs

tempo_inst = tempo_picos[:-1] + (numpy.diff(tempo_picos) / 2) 

# gráfico temporal
plt.figure(figsize=(12, 4))
plt.plot(tempo_inst, FC_instantanea, 'o-', markersize=3, label='FC Instantânea')
plt.axhline(y=mediaBPM, color='r', linestyle='--', label=f'Média: {mediaBPM:.2f} BPM')
plt.title('Variação da Frequência Cardíaca ao Longo do Registro (Tacograma)')
plt.xlabel('Tempo (s)')
plt.ylabel('Frequência Cardíaca (BPM)')
plt.xlim(0, 20)
plt.savefig(f"{registro_ID} - varia_freq.png")
plt.close()

# histograma
plt.figure(figsize=(8, 5))
plt.hist(intervalosRR_ms, bins=50, color='skyblue', edgecolor='black')
plt.title('Histograma da Distribuição dos Intervalos RR')
plt.xlabel('Intervalo RR (ms)')
plt.ylabel('Contagem (Frequência)')
plt.savefig(f"{registro_ID} - hist_intervalos.png")
plt.close()

# ------------------------- METRICAS DE DESEMPENHO DE DETECÇÃO ---------------------------
anotacoes_registro = wfdb.rdann(registro_ID, 'atr', pn_dir='mitdb')
simbolos_qrs = ['N', 'L', 'R', 'A', 'a', 'J', 'S', 'V', 'F', 'e', 'j', '/', 'E', 'f', 'Q']
picos_r_referencia = anotacoes_registro.sample[numpy.isin(anotacoes_registro.symbol, simbolos_qrs)]

tolerancia_amostras = (0.100 * fs) 

comparacao = compare_annotations(
    picos_r_referencia,
    picosR,
    tolerancia_amostras
)

verdadeiro_positivo = comparacao.tp
falso_negativo = comparacao.fn
falso_positivo = comparacao.fp

Sensibilidade = (verdadeiro_positivo / (verdadeiro_positivo + falso_negativo)) * 100
preditivo_pos = (verdadeiro_positivo / (verdadeiro_positivo + falso_positivo)) * 100
print("\n==================== Tabela 2 ======================")
print("|| Métricas de Desempenho da Detecção de Picos R  ||")
print("||                                                ||")
print(f"|| Total de Batimentos de Referência  {len(picos_r_referencia):4d}        ||")
print(f"|| Verdadeiros Positivos (TP)         {verdadeiro_positivo:4d}        ||")
print(f"|| Falsos Negativos (FN)              {falso_negativo:4d}        ||")
print(f"|| Falsos Positivos (FP)              {falso_positivo:4d}        ||")
print("|| ---------------------------------------------- ||")
print(f"|| Sensibilidade (Se)                 {Sensibilidade:3.2f} %     ||")
print(f"|| Valor Preditivo Positivo (PPV)     {preditivo_pos:3.2f} %     ||")
print("====================================================")