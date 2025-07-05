# Instalação das dependências (execute no terminal)
# pip install ultralytics labelme2yolo opencv-python

import os
from ultralytics import YOLO
import shutil

# 1. Preparação dos dados do LabelMe para YOLO
def convert_labelme_to_yolo(labelme_folder, output_folder):
    from labelme2yolo import labelme2yolo
    labelme2yolo.convert(labelme_folder, output_folder)

# Diretórios (ajuste conforme seu caso)
labelme_data = "path/to/labelme/data"
yolo_data = "yolo_dataset"

# Converter anotações
convert_labelme_to_yolo(labelme_data, yolo_data)

# 2. Estrutura de diretórios para YOLO
os.makedirs(os.path.join(yolo_data, "train/images"), exist_ok=True)
os.makedirs(os.path.join(yolo_data, "train/labels"), exist_ok=True)
os.makedirs(os.path.join(yolo_data, "val/images"), exist_ok=True)
os.makedirs(os.path.join(yolo_data, "val/labels"), exist_ok=True)

# 3. Dividir dados em treino/validação (80/20)
from sklearn.model_selection import train_test_split

all_images = [f for f in os.listdir(os.path.join(yolo_data, "images")) if f.endswith(('.jpg', '.png'))]
train_files, val_files = train_test_split(all_images, test_size=0.2, random_state=42)

# Mover arquivos para pastas correspondentes
for file in train_files:
    # Mover imagens
    shutil.move(os.path.join(yolo_data, "images", file),
                os.path.join(yolo_data, "train/images", file))
    # Mover labels
    label_file = os.path.splitext(file)[0] + '.txt'
    shutil.move(os.path.join(yolo_data, "labels", label_file),
                os.path.join(yolo_data, "train/labels", label_file))

for file in val_files:
    # Mover imagens
    shutil.move(os.path.join(yolo_data, "images", file),
                os.path.join(yolo_data, "val/images", file))
    # Mover labels
    label_file = os.path.splitext(file)[0] + '.txt'
    shutil.move(os.path.join(yolo_data, "labels", label_file),
                os.path.join(yolo_data, "val/labels", label_file))

# 4. Criar arquivo dataset.yaml
yaml_content = f"""
path: {os.path.abspath(yolo_data)}
train: train/images
val: val/images

names:
  0: classe1  # Substitua pelas suas classes
  1: classe2
  2: classe3
"""

with open(os.path.join(yolo_data, "dataset.yaml"), "w") as f:
    f.write(yaml_content)

# 5. Treinar o modelo YOLOv8
model = YOLO("yolov8n.pt")  # Carrega um modelo pré-treinado pequeno

results = model.train(
    data=os.path.join(yolo_data, "dataset.yaml"),
    epochs=100,
    imgsz=640,
    batch=8,
    name='my_custom_yolo'
)

# 6. Avaliar o modelo
metrics = model.val()
print(metrics.box.map)  # mAP50-95

# 7. Fazer previsões em novas imagens
results = model.predict("test_image.jpg", save=True, conf=0.5)