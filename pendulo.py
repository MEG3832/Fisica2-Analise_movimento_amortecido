import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Encontra os tempos e as coordenadas em cada frame
def encontraDados(video, fps):
    dados = []
    frame_id = 0
    while(True):
        leu, frame = video.read()
        if not leu:
            break

        tempo = frame_id / fps

        # Converte para escala de cinza
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplica limiar para binarizar (preto e branco invertido)
        _, limiar = cv2.threshold(cinza, 100, 255, cv2.THRESH_BINARY_INV)

        # Encontra contornos
        contornos, _ = cv2.findContours(limiar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contornos:
            maior = max(contornos, key=cv2.contourArea)
            M = cv2.moments(maior)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dados.append((tempo, (cx, cy)))

        frame_id += 1

    return dados

# Função da onda do movimento amortecido
def modeloMA(t, A, w, phi, b):
    return A * np.exp(-b * t) * np.sin(w * t + phi) 

def criaGraficos(dados):
    tempos = [d[0] for d in dados]
    x_vals = [d[1][0] for d in dados]
    y_vals = [d[1][1] for d in dados]

    # Gráfico da posição X
    plt.figure(figsize=(10, 5))
    plt.plot(tempos, x_vals, 'bo', label='x(t) dados')

    # Ajuste da curva x(t)
    A0 = (max(x_vals) - min(x_vals)) / 2
    w0 = 2 * np.pi / (tempos[-1] if tempos[-1] > 0 else 1)  # tem esse if-else para o caso de t ser 0 ou próximo
    phi0 = 0
    b0 = 0.1
    p0 = [A0, w0, phi0, b0]

    try:
        parametros_otimizados, _ = curve_fit(modeloMA, tempos, x_vals, p0) # Encontra os melhores valores de w, phi e A para que meus dados sejam um MHS
        x_ajustada = modeloMA(np.array(tempos), *parametros_otimizados)    # Recalcula as posições em x a partir dos tempos e dos parâmetros (w, phi e A) otimizados
        plt.plot(tempos, x_ajustada, 'r-', label='Ajuste OHM')  # Plota o gráfico de x pelo tempo novamente, potém com os valores otimizados
    except RuntimeError:
        print("Não foi possível ajustar a curva.")

    plt.xlabel('Tempo (s)')
    plt.ylabel('Posição X (pixels)')
    plt.title('Posição X e ajuste OHM')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("graficoX_ajustado.png")

    # Gráfico da posição Y
    plt.figure(figsize=(10, 5))
    plt.plot(tempos, y_vals, 'go', label='y(t) dados')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Posição Y (pixels)')
    plt.title('Posição Y pelo tempo')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("graficoY.png")

    return parametros_otimizados

video = cv2.VideoCapture("video_pendulo.mp4")

# Ver quantos frames por segundo tem o vídeo
fps = video.get(cv2.CAP_PROP_FPS)

dados = encontraDados(video, fps)

A, w, phi, b = criaGraficos(dados) # Faz as impressões finais das variáveis
print(f"Amplitude = {A:.2f}")
print(f"Frequencia ângular = {w:.2f}")
print(f"Ângulo = {phi:.2f}")
print(f"Coeficiente de amortecimento = {b:.2f}")

Q = w / (2 * b)
print(f"Fator de qualidade = {Q:.2f}")

video.release()
