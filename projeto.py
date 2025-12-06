import wfdb                                 # Biblioteca especializada para ler dados do PhysioNet, base de dados que foi coletada as ECGs (MIT-BIH)
import numpy
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt   # Bibliotecas de filtragem  
import neurokit2                            # Biblioteca para detecção de picos R e análise de VFC/HRV
import ts2vg                                # Biblioteca auxiliar para método de detecção de picos R

#Adicional
#análise espectral de HRV para ilustrar o equilíbrio autonômico e discutir implicações fisiológicas

# ------------------------ RECEBER AS INFORMAÇÕES DO BANCO DE DADOS ---------------------------------
taxa_amostragem = 360   # amostragem padrão do banco de dados
canal = 0               # banco de dados foi dividido em dois canais, um deles foi escolhido
registro_ID = '121'     # são 48 registros, escolhido o primeiro

record = wfdb.rdrecord(registro_ID, pn_dir= 'mitdb')
sinal_bruto = record.p_signal[:, canal]
fs = record.fs

plt.figure(figsize=(12, 4))
plt.plot(numpy.arange(len(sinal_bruto)) / fs, sinal_bruto)
plt.title(f'Sinal ECG Bruto (Registro {registro_ID})')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude (mV)')
plt.savefig("plot_sinal.png")
plt.close

# -------------------------------- PRÉ PROCESSAMENTO --------------------------------------------------
#filtragem passa-banda (0,5Hz a 40Hz) para eliminar ruídos de baixa frequencia e interferência de rede elétrica (50/60Hz)
freq_baixa = 0.5
freq_alta = 40
ordem_filtro = 5           # número de polos do filtro, inclinação com que o filtro atenua as frequencias fora da banda de passagem, (4, 5 ou 6) são as mais equilibradas para ECG

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

plt.figure(figsize=(12, 4))
plt.plot(numpy.arange(len(sinal_bruto)) / fs, sinal_bruto, label='Bruto', alpha=0.5)
plt.plot(numpy.arange(len(sinal_filtrado)) / fs, sinal_filtrado, label='Filtrado')
plt.title('Comparação: Sinal Bruto vs. Sinal Filtrado (Passa-Banda)')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude (mV)')
plt.xlim(0, 10)
plt.savefig("comparar_sinal.png")
plt.close

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
plt.savefig("tentar_neurokit.png")
plt.close

#--------------------------------- VARIABILIDADE DA FREQUENCIA CARDIACA ---------------------------------
# diferencas entre amostras para cálculo de intervalos RR
diferencas = numpy.diff(picosR)
intervalosRR_FC = diferencas/fs             # intervalos RR em segundos para FC
intervalosRR_hist = intervalosRR_FC * 1000  # intervalos RR em milissegundos para histograma

# cálculo da frequencia cardíaca instantânea
FC_instantanea = 60/intervalosRR_FC

# média e desvio padrão dos intervalos RR
mediaRR = numpy.mean(intervalosRR_hist)
desvioRR = numpy.std(intervalosRR_hist)

#média de batimentos por minuto (BPM)
mediaBPM = numpy.mean(FC_instantanea)

print("\n--- Tabela de Resumo dos Intervalos RR ---")
print(f"Média dos Intervalos RR: {mediaRR:.2f} ms")
print(f"Desvio Padrão dos Intervalos RR (SDNN): {desvioRR:.2f} ms")
print(f"Frequência Cardíaca Média (BPM): {mediaBPM:.2f} BPM")

# -------------------------- CONSTRUÇÃO DOS GRÁFICOS TEMPORAIS E HISTOGRAMA ----------------------------
# marca o instante exato de cada pico R detectado
tempo_picos = picosR / fs

# vetor com quantidade de picos - 1 contendo os meios dos intervalos entre as batidas para construir o tacograma/gráfico temporal
tempo_inst = tempo_picos[:-1] + (numpy.diff(tempo_picos) / 2) 

plt.figure(figsize=(12, 4))
plt.plot(tempo_inst, FC_instantanea, 'o-', markersize=3, label='FC Instantânea')
plt.axhline(y=mediaBPM, color='r', linestyle='--', label=f'Média: {mediaBPM:.2f} BPM')
plt.title('Variação da Frequência Cardíaca ao Longo do Registro (Tacograma)')
plt.xlabel('Tempo (s)')
plt.ylabel('Frequência Cardíaca (BPM)')
plt.savefig("varia_freq.png")
plt.close

# -------------------------- FILTRO PARA REFINAR OS FALSOS PICOS ----------------------------
largura_janela = 10
limiar_percentual = 0.10 # 20% de tolerância

kernel_media_movel = numpy.ones(largura_janela) / largura_janela
MM = numpy.convolve(intervalosRR_hist, kernel_media_movel, mode='same')
rr_corrigido = intervalosRR_hist.copy()

idx_inicial = int((largura_janela - 1) / 2) 
idx_final = len(rr_corrigido) - idx_inicial

for i in range(idx_inicial, idx_final):
    diferenca_absoluta = numpy.abs(rr_corrigido[i] - MM[i])
    limiar = limiar_percentual * MM[i]
    
    if diferenca_absoluta > limiar:
        rr_corrigido[i] = MM[i]

FC_instantanea_limpa = 60000 / rr_corrigido # 60 * 1000 / RR_ms

# Média e DP corrigidos (SDNN)
media_corrigida_ms = numpy.mean(rr_corrigido)
desvio_corrigido_ms = numpy.std(rr_corrigido)
bpm_nova = numpy.mean(FC_instantanea_limpa)

print("\n--- Resultados Após Filtro Adaptativo (Limiar 20%) ---")
print(f"Média dos Intervalos RR Corrigidos: {media_corrigida_ms:.2f} ms")
print(f"SDNN (Desvio Padrão): {desvio_corrigido_ms:.2f} ms")

plt.figure(figsize=(12, 4))
plt.plot(tempo_inst, FC_instantanea_limpa, 'o-', markersize=3, label='FC Instantânea')
plt.axhline(y=bpm_nova, color='r', linestyle='--', label=f'Média: {bpm_nova:.2f} BPM')
plt.title('Variação da Frequência Cardíaca ao Longo do Registro (Tacograma)')
plt.xlabel('Tempo (s)')
plt.ylabel('Frequência Cardíaca (BPM)')
plt.savefig("varia_freq_limpo.png")
plt.close

plt.figure(figsize=(8, 5))
plt.hist(rr_corrigido, bins=50, color='skyblue', edgecolor='black')
plt.title('Histograma da Distribuição dos Intervalos RR')
plt.xlabel('Intervalo RR (ms)')
plt.ylabel('Contagem (Frequência)')
plt.savefig("hist_intervalos.png")
plt.close