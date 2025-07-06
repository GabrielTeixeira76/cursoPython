import numpy as np
import os
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import matplotlib.pyplot as plt

class SistemaRecomendacaoImagens:
    def __init__(self, diretorio_imagens):
        """
        Inicializa o sistema de recomendação
        
        Args:
            diretorio_imagens (str): Caminho para o diretório contendo as imagens do catálogo
        """
        self.diretorio_imagens = diretorio_imagens
        self.lista_imagens = []
        self.caracteristicas = []
        self.modelo = self._carregar_modelo()
    
    def _carregar_modelo(self):
        """Carrega o modelo VGG16 pré-treinado (sem as camadas fully connected)"""
        return VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    def _preprocessar_imagem(self, caminho_imagem):
        """
        Pré-processa uma imagem para ser compatível com o modelo VGG16
        
        Args:
            caminho_imagem (str): Caminho para o arquivo de imagem
            
        Returns:
            numpy.array: Imagem pré-processada
        """
        img = image.load_img(caminho_imagem, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expandido = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array_expandido)
    
    def _extrair_caracteristicas(self, caminho_imagem):
        """
        Extrai características visuais de uma imagem usando o modelo VGG16
        
        Args:
            caminho_imagem (str): Caminho para o arquivo de imagem
            
        Returns:
            numpy.array: Vetor de características da imagem
        """
        img_preprocessada = self._preprocessar_imagem(caminho_imagem)
        caracteristicas = self.modelo.predict(img_preprocessada)
        return caracteristicas.flatten()
    
    def carregar_catalogo(self):
        """Carrega todas as imagens do catálogo e extrai suas características"""
        self.lista_imagens = [os.path.join(self.diretorio_imagens, f) 
                            for f in os.listdir(self.diretorio_imagens) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Carregando e processando {len(self.lista_imagens)} imagens...")
        for caminho_imagem in tqdm(self.lista_imagens):
            try:
                caracteristicas = self._extrair_caracteristicas(caminho_imagem)
                self.caracteristicas.append(caracteristicas)
            except Exception as e:
                print(f"Erro ao processar {caminho_imagem}: {str(e)}")
                continue
    
    def recomendar(self, caminho_imagem_consulta, top_n=5):
        """
        Encontra as imagens mais similares no catálogo
        
        Args:
            caminho_imagem_consulta (str): Caminho para a imagem de consulta
            top_n (int): Número de recomendações a retornar
            
        Returns:
            list: Lista de tuplas (caminho_imagem, score_similaridade)
        """
        # Extrair características da imagem de consulta
        try:
            caracteristicas_consulta = self._extrair_caracteristicas(caminho_imagem_consulta)
        except Exception as e:
            print(f"Erro ao processar imagem de consulta: {str(e)}")
            return []
        
        # Calcular similaridade com todas as imagens do catálogo
        similaridades = cosine_similarity(
            [caracteristicas_consulta], 
            self.caracteristicas
        )[0]
        
        # Obter os índices das imagens mais similares
        indices_mais_similares = np.argsort(similaridades)[-top_n-1:-1][::-1]
        
        # Retornar caminhos das imagens e scores de similaridade
        return [(self.lista_imagens[i], similaridades[i]) 
               for i in indices_mais_similares]
    
    def visualizar_recomendacoes(self, caminho_imagem_consulta, top_n=5):
        """
        Exibe a imagem de consulta e as imagens recomendadas
        
        Args:
            caminho_imagem_consulta (str): Caminho para a imagem de consulta
            top_n (int): Número de recomendações a exibir
        """
        recomendacoes = self.recomendar(caminho_imagem_consulta, top_n)
        
        if not recomendacoes:
            print("Nenhuma recomendação disponível.")
            return
        
        plt.figure(figsize=(15, 8))
        
        # Exibir imagem de consulta
        plt.subplot(1, top_n+1, 1)
        img_consulta = Image.open(caminho_imagem_consulta)
        plt.imshow(img_consulta)
        plt.title("Imagem Consulta")
        plt.axis('off')
        
        # Exibir imagens recomendadas
        for i, (caminho, score) in enumerate(recomendacoes, 2):
            plt.subplot(1, top_n+1, i)
            img = Image.open(caminho)
            plt.imshow(img)
            plt.title(f"Similaridade: {score:.2f}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Exemplo de uso
if __name__ == "__main__":
    # 1. Inicializar o sistema com o diretório das imagens
    sistema = SistemaRecomendacaoImagens("caminho/para/seu/catalogo")
    
    # 2. Carregar e processar todas as imagens do catálogo
    sistema.carregar_catalogo()
    
    # 3. Fazer uma consulta e visualizar resultados
    imagem_consulta = "caminho/para/imagem/consulta.jpg"
    sistema.visualizar_recomendacoes(imagem_consulta, top_n=5)import numpy as np
import os
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import matplotlib.pyplot as plt

class SistemaRecomendacaoImagens:
    def __init__(self, diretorio_imagens):
        """
        Inicializa o sistema de recomendação
        
        Args:
            diretorio_imagens (str): Caminho para o diretório contendo as imagens do catálogo
        """
        self.diretorio_imagens = diretorio_imagens
        self.lista_imagens = []
        self.caracteristicas = []
        self.modelo = self._carregar_modelo()
    
    def _carregar_modelo(self):
        """Carrega o modelo VGG16 pré-treinado (sem as camadas fully connected)"""
        return VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    def _preprocessar_imagem(self, caminho_imagem):
        """
        Pré-processa uma imagem para ser compatível com o modelo VGG16
        
        Args:
            caminho_imagem (str): Caminho para o arquivo de imagem
            
        Returns:
            numpy.array: Imagem pré-processada
        """
        img = image.load_img(caminho_imagem, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expandido = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array_expandido)
    
    def _extrair_caracteristicas(self, caminho_imagem):
        """
        Extrai características visuais de uma imagem usando o modelo VGG16
        
        Args:
            caminho_imagem (str): Caminho para o arquivo de imagem
            
        Returns:
            numpy.array: Vetor de características da imagem
        """
        img_preprocessada = self._preprocessar_imagem(caminho_imagem)
        caracteristicas = self.modelo.predict(img_preprocessada)
        return caracteristicas.flatten()
    
    def carregar_catalogo(self):
        """Carrega todas as imagens do catálogo e extrai suas características"""
        self.lista_imagens = [os.path.join(self.diretorio_imagens, f) 
                            for f in os.listdir(self.diretorio_imagens) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Carregando e processando {len(self.lista_imagens)} imagens...")
        for caminho_imagem in tqdm(self.lista_imagens):
            try:
                caracteristicas = self._extrair_caracteristicas(caminho_imagem)
                self.caracteristicas.append(caracteristicas)
            except Exception as e:
                print(f"Erro ao processar {caminho_imagem}: {str(e)}")
                continue
    
    def recomendar(self, caminho_imagem_consulta, top_n=5):
        """
        Encontra as imagens mais similares no catálogo
        
        Args:
            caminho_imagem_consulta (str): Caminho para a imagem de consulta
            top_n (int): Número de recomendações a retornar
            
        Returns:
            list: Lista de tuplas (caminho_imagem, score_similaridade)
        """
        # Extrair características da imagem de consulta
        try:
            caracteristicas_consulta = self._extrair_caracteristicas(caminho_imagem_consulta)
        except Exception as e:
            print(f"Erro ao processar imagem de consulta: {str(e)}")
            return []
        
        # Calcular similaridade com todas as imagens do catálogo
        similaridades = cosine_similarity(
            [caracteristicas_consulta], 
            self.caracteristicas
        )[0]
        
        # Obter os índices das imagens mais similares
        indices_mais_similares = np.argsort(similaridades)[-top_n-1:-1][::-1]
        
        # Retornar caminhos das imagens e scores de similaridade
        return [(self.lista_imagens[i], similaridades[i]) 
               for i in indices_mais_similares]
    
    def visualizar_recomendacoes(self, caminho_imagem_consulta, top_n=5):
        """
        Exibe a imagem de consulta e as imagens recomendadas
        
        Args:
            caminho_imagem_consulta (str): Caminho para a imagem de consulta
            top_n (int): Número de recomendações a exibir
        """
        recomendacoes = self.recomendar(caminho_imagem_consulta, top_n)
        
        if not recomendacoes:
            print("Nenhuma recomendação disponível.")
            return
        
        plt.figure(figsize=(15, 8))
        
        # Exibir imagem de consulta
        plt.subplot(1, top_n+1, 1)
        img_consulta = Image.open(caminho_imagem_consulta)
        plt.imshow(img_consulta)
        plt.title("Imagem Consulta")
        plt.axis('off')
        
        # Exibir imagens recomendadas
        for i, (caminho, score) in enumerate(recomendacoes, 2):
            plt.subplot(1, top_n+1, i)
            img = Image.open(caminho)
            plt.imshow(img)
            plt.title(f"Similaridade: {score:.2f}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Exemplo de uso
if __name__ == "__main__":
    # 1. Inicializar o sistema com o diretório das imagens
    sistema = SistemaRecomendacaoImagens("caminho/para/seu/catalogo")
    
    # 2. Carregar e processar todas as imagens do catálogo
    sistema.carregar_catalogo()
    
    # 3. Fazer uma consulta e visualizar resultados
    imagem_consulta = "caminho/para/imagem/consulta.jpg"
    sistema.visualizar_recomendacoes(imagem_consulta, top_n=5)