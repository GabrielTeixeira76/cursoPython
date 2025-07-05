import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_to_grayscale(image_path):
    # Carrega a imagem colorida
    img_color = cv2.imread(image_path)
    
    # Converte para RGB (OpenCV carrega como BGR por padrão)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    
    # Converte para tons de cinza usando a média dos canais R, G, B
    img_gray = np.mean(img_color, axis=2).astype(np.uint8)
    
    return img_color, img_gray

def convert_to_binary(img_gray, threshold=128):
    # Aplica limiarização simples
    _, img_binary = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
    return img_binary

def display_images(original, gray, binary):
    # Configura a exibição das imagens
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Imagem em Cinza')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(binary, cmap='gray')
    plt.title('Imagem Binária')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Exemplo de uso
if __name__ == "__main__":
    # Substitua pelo caminho da sua imagem
    image_path = 'lena.png'  # Você precisa ter a imagem Lena no diretório
    
    # Converte a imagem
    original, gray = convert_to_grayscale(image_path)
    binary = convert_to_binary(gray)
    
    # Exibe os resultados
    display_images(original, gray, binary)
    
    # Salva as imagens resultantes (opcional)
    cv2.imwrite('lena_gray.jpg', gray)
    cv2.imwrite('lena_binary.jpg', binary)