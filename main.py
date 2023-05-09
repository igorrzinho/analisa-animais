import cv2
from matplotlib import pyplot as plt

imagem1 = cv2.imread('img/img1.jpg', 0)  # Carrega a imagem 1 em escala de cinza
imagem2 = cv2.imread('img/img7.jpg', 0)  # Carrega a imagem 2 em escala de cinza

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(imagem1, None)
kp2, des2 = orb.detectAndCompute(imagem2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

limite_distancia = 50  # Valor arbitrário, você pode ajustá-lo conforme necessário

melhores_correspondencias = [match for match in matches if match.distance < limite_distancia]

porcentagem_correspondencias = len(melhores_correspondencias) / len(matches) * 100

if porcentagem_correspondencias >= 6 :  # Valor arbitrário, você pode ajustá-lo conforme necessário
    print("As imagens mostram o mesmo animal.")
else:
    print("As imagens mostram animais diferentes.")


