import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# Carregar os dados
data = pd.read_csv('face_landmarks.csv')

# Selecionar apenas as coordenadas x e y dos landmarks
X = data.drop('face_id', axis=1).iloc[:, :936]  # Seleciona apenas as 936 primeiras colunas (x e y de 468 landmarks)
y = data['face_id']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar um modelo SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Fazer previsões e calcular a acurácia
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy * 100:.2f}%')

# Para a fase de previsão, extraímos os landmarks da imagem e usamos apenas as coordenadas x e y
import cv2
import mediapipe as mp

# Crie uma instância do detector de landmarks
mp_face_mesh = mp.solutions.face_mesh
detector = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Carregue a imagem de entrada
input_image = cv2.imread('Robert.jpg')

# Converta a imagem para RGB (mediapipe trabalha com imagens RGB)
input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Detecte os landmarks na imagem de entrada
results = detector.process(input_image_rgb)
if results.multi_face_landmarks:
    input_landmarks = results.multi_face_landmarks[0].landmark

    # Extraia as coordenadas dos landmarks (apenas x e y)
    input_landmarks_list = [[lm.x, lm.y] for lm in input_landmarks]

    # Transforme em um array numpy e redimensione para 1D
    input_features = np.array(input_landmarks_list).reshape(1, -1)

    # Faça a previsão com o modelo treinado
    prediction = model.predict(input_features)

    # Verifique a saída do modelo
    if prediction == 'face_1':
        print("A imagem de entrada corresponde à face 1.")
    else:
        print("A imagem de entrada não corresponde à face 1.")
