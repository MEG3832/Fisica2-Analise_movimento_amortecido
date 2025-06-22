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

        # Binariza (brilho < 100 vira branco (255), e o resto vira preto (0))
        _, limiar = cv2.threshold(cinza, 100, 255, cv2.THRESH_BINARY_INV)

        # Encontra contornos
        contornos, _ = cv2.findContours(limiar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calcula as coordenadas do centro da massa
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
def modeloMHA(t, A, b, w, phi, C):
    return A * np.exp((-b * t) / (2 * massa)) * np.cos(w * t + phi) + C

# Cria os dois gráficos e retorna os dados
def criaGraficos(dados):
    # Listas com cada tempo, deslocamento em x e deslocamento em y
    tempos = np.array([d[0] for d in dados])
    x_vals = np.array([d[1][0] for d in dados])
    y_vals = np.array([d[1][1] for d in dados])

    # Cria gráfico da posição X pelo tempo
    plt.figure(figsize=(12, 6))
    plt.plot(tempos, x_vals, color='blue', linewidth=2)
    plt.title('Posição X da massa em função do tempo')
    plt.xlabel('Tempo (s)')
    plt.ylabel('X (px)')
    plt.grid(True)
    plt.savefig("grafico_x.png")
    plt.close()

    # Cria o gráfico da posição Y pelo tempo
    plt.figure(figsize=(12, 6))
    plt.plot(tempos, y_vals, color='green', linewidth=2)
    plt.title('Posição Y da massa em função do tempo')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Y (px)')
    plt.grid(True)
    plt.savefig("grafico_y.png")
    plt.close()

    return tempos, x_vals

# Ajusta a curva para os dados de x(t)
def ajustaCurva(tempos, x_vals):
    global massa
    massa = 0.190
    A0 = (max(x_vals) - min(x_vals)) / 2
    b0 = 0.005
    T0 = 3.25
    w0 = 2*np.pi/T0
    phi0 = 0
    C0 = np.mean(x_vals)
    p0 = [A0, b0, w0, phi0, C0]

    try:
        # Obtém os valores ajustados de A, b, w, phi, C segundo o que eu adquiri experimentalmente, meus chutes iniciais e a fórmula do MHA
        popt, _ = curve_fit(modeloMHA, tempos, x_vals, p0=p0, maxfev=10000)
        A, b, w, phi, C = popt
        gamma = b / (2*massa)
        Q = w / (2*gamma)

        # Imprime os valores
        print("\nPARÂMETROS AJUSTADOS")
        print(f"A = {A:.4f} pixels")
        print(f"b = {b:.6f}")
        print(f"w = {w:.4f} rad/s")
        print(f"phi = {phi:.4f} rad")
        print(f"C = {C:.4f} pixels")
        print(f"Q = {Q:.2f}")

        # Gera um gráfico de X pelo tempo a partir das posições com os parâmetros ajustados
        x_ajustada = modeloMHA(tempos, *popt)

        plt.figure(figsize=(12, 6))
        plt.plot(tempos, x_vals, 'bo', label='Dados', markersize=3)
        plt.plot(tempos, x_ajustada, 'r-', label='Ajuste', linewidth=2)
        plt.title('Ajuste de x(t) pelo Modelo de MHA')
        plt.xlabel('Tempo (s)')
        plt.ylabel('X (px)')
        plt.grid(True)
        plt.legend()
        plt.savefig("ajuste_xt.png")
        plt.close()

    except RuntimeError:
        print("Erro: ajuste falhou.")

# Main
video = cv2.VideoCapture("video_pendulo.mp4")

fps = video.get(cv2.CAP_PROP_FPS)

dados = encontraDados(video, fps)

tempos, x_vals = criaGraficos(dados)

ajustaCurva(tempos, x_vals)

video.release()
