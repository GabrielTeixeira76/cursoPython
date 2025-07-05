# %% [markdown]
# # Transfer Learning para Classificação de Gatos e Cachorros
# 
# Este notebook demonstra como usar Transfer Learning com uma rede pré-treinada para classificar imagens de gatos e cachorros.

# %% [markdown]
# ## Configuração Inicial

# %%
# Importar bibliotecas necessárias
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile

# Verificar versão do TensorFlow
print("TensorFlow version:", tf.__version__)

# %% [markdown]
# ## Preparação dos Dados

# %%
# Baixar e extrair o dataset (executar apenas uma vez)
!wget --no-check-certificate \
    https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip \
    -O /tmp/cats-and-dogs.zip

local_zip = '/tmp/cats-and-dogs.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

# %%
# Criar diretórios organizados para o dataset
base_dir = '/tmp/cats_and_dogs'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Criar subdiretórios para cada classe
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# Criar os diretórios
os.makedirs(train_cats_dir, exist_ok=True)
os.makedirs(train_dogs_dir, exist_ok=True)
os.makedirs(validation_cats_dir, exist_ok=True)
os.makedirs(validation_dogs_dir, exist_ok=True)

# %%
# Mover imagens para os diretórios corretos (executar apenas uma vez)
import shutil

# Função para copiar imagens
def copy_images(source_dir, target_dir, start_idx, end_idx):
    fnames = [f'{i}.jpg' for i in range(start_idx, end_idx)]
    for fname in fnames:
        src = os.path.join(source_dir, fname)
        dst = os.path.join(target_dir, fname)
        try:
            shutil.copyfile(src, dst)
        except FileNotFoundError:
            pass

# Copiar imagens de gatos
copy_images('/tmp/PetImages/Cat', train_cats_dir, 0, 1000)
copy_images('/tmp/PetImages/Cat', validation_cats_dir, 1000, 1400)

# Copiar imagens de cachorros
copy_images('/tmp/PetImages/Dog', train_dogs_dir, 0, 1000)
copy_images('/tmp/PetImages/Dog', validation_dogs_dir, 1000, 1400)

# Verificar quantidade de imagens em cada diretório
print('Total de imagens de gatos para treino:', len(os.listdir(train_cats_dir)))
print('Total de imagens de cachorros para treino:', len(os.listdir(train_dogs_dir))))
print('Total de imagens de gatos para validação:', len(os.listdir(validation_cats_dir))))
print('Total de imagens de cachorros para validação:', len(os.listdir(validation_dogs_dir))))

# %% [markdown]
# ## Pré-processamento dos Dados

# %%
# Configurar geradores de dados com aumento de dados (data augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

# Configurar fluxo de imagens de treino e validação
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

# %% [markdown]
# ## Modelo com Transfer Learning

# %%
# Carregar modelo pré-treinado (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(150, 150, 3),
    include_top=False,
    weights='imagenet')

# Congelar a base convolucional
base_model.trainable = False

# %%
# Construir modelo completo
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

# Compilar o modelo
model.compile(optimizer=keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Resumo do modelo
model.summary()

# %% [markdown]
# ## Treinamento do Modelo

# %%
# Treinar o modelo
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50)

# %% [markdown]
# ## Avaliação do Modelo

# %%
# Plotar gráficos de acurácia e loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# %% [markdown]
# ## Fine Tuning

# %%
# Descongelar as últimas camadas da base
base_model.trainable = True

# Congelar todas as camadas exceto as últimas 5
for layer in base_model.layers[:-5]:
    layer.trainable = False

# Recompilar o modelo
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Resumo do modelo após fine tuning
model.summary()

# %%
# Continuar treinamento com fine tuning
history_fine = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=50)

# %%
# Plotar resultados após fine tuning
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# %% [markdown]
# ## Salvando o Modelo

# %%
# Salvar o modelo treinado
model.save('/tmp/cats_and_dogs_model.h5')

# %% [markdown]
# ## Testando o Modelo com Novas Imagens

# %%
# Função para carregar e preparar imagem
def load_and_prepare_image(img_path, target_size=(150, 150)):
    img = keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar dimensão do batch
    img_array /= 255.0  # Normalizar como no treino
    return img_array

# %%
# Testar com uma imagem de exemplo (substitua pelo caminho da sua imagem)
test_image_path = '/tmp/PetImages/Cat/100.jpg'  # Altere para sua imagem
img_array = load_and_prepare_image(test_image_path)

# Fazer predição
prediction = model.predict(img_array)
print(f'Probabilidade de ser cachorro: {prediction[0][0]:.2f}')

# Mostrar a imagem
img = plt.imread(test_image_path)
plt.imshow(img)
plt.axis('off')
plt.show()