import cv2
import numpy as np
import glob
import yaml

# Configurações
CHECKERBOARD = (8,5)  # Número de cantos INTERNOS (linhas-1, colunas-1)
SQUARE_SIZE = 0.024   # Tamanho do quadrado em metros

# Preparar pontos do objeto 3D
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2) * SQUARE_SIZE

# Armazenar pontos 3D e 2D
objpoints = []  # Pontos 3D no mundo real
imgpoints = []  # Pontos 2D no plano da imagem

# Capturar imagens
cap = cv2.VideoCapture(0)
img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret:
        # Refinar detecção
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        # Desenhar e mostrar
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
    
    cv2.imshow('Calibracao - Pressione "c" para capturar, "q" para sair', frame)
    
    key = cv2.waitKey(1)
    if key == ord('c') and ret:
        objpoints.append(objp)
        imgpoints.append(corners2)
        img_count +=1
        print(f"Capturada imagem {img_count}/20")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calibrar câmera
if len(objpoints) >= 10:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Converter para o formato desejado
    calibration_data = {
        'camera_matrix': {
            'rows': int(mtx.shape[0]),
            'cols': int(mtx.shape[1]),
            'data': mtx.flatten().tolist()
        },
        'dist_coeffs': dist.flatten().tolist(),
        'resolution': {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
    }
    
    # Salvar em YAML com formatação específica
    with open("config/camera_calibration.yaml", 'w') as f:
        yaml.dump(calibration_data, f, default_flow_style=None, sort_keys=False)
    
    print(f"Calibração concluída!\nErro RMS: {ret}\nSalvo em camera_calibration.yaml")
else:
    print("Número insuficiente de imagens válidas (mínimo 10)")