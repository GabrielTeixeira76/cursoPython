import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 1. Carregar modelos
face_detector = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
classification_model = load_model('meu_modelo_classificacao.h5')

# 2. Definir classes (ajuste conforme seus personagens)
CLASSES = ["sheldon", "leonard", "penny", "raj", "amy", "bernadette"]

# 3. Função para pré-processamento da imagem
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# 4. Carregar imagem de entrada
image = cv2.imread('input_image.jpg')
(h, w) = image.shape[:2]

# 5. Detecção de faces
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
face_detector.setInput(blob)
detections = face_detector.forward()

# 6. Loop pelas detecções
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    
    # Filtrar detecções fracas
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        # Extrair ROI (Região de Interesse)
        face = image[startY:endY, startX:endX]
        
        # Classificação
        processed_face = preprocess_image(face)
        predictions = classification_model.predict(processed_face)[0]
        label_idx = np.argmax(predictions)
        label = CLASSES[label_idx]
        confidence = predictions[label_idx]
        
        # Desenhar resultado na imagem
        text = f"{label} ({confidence:.2f})"
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# 7. Mostrar resultado
cv2.imshow("Reconhecimento Facial", image)
cv2.waitKey(0)
cv2.destroyAllWindows()