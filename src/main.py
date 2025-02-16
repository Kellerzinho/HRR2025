# src/main.py

import time
import math
import cv2

# Módulos desenvolvidos
from core.camera import Camera
from core.vision import VisionSystem
from core.perception import Perception
from core.slam import MonteCarloLocalization
from core.strategy import Strategy
from core.navigation import Navigator
from core.control_pid import MotionController
from utils.config_loader import load_config


def main():
    """
    Exemplo de loop principal que integra:
      - camera.py      (captura de frames)
      - vision.py      (YOLOv8 para segmentação de landmarks)
      - perception.py  (fusão de IMU + conversão de coordenadas)
      - slam.py        (Monte Carlo Localization)
    """

    # -------------------------------------------------------------
    # 1) CONFIGURAÇÕES INICIAIS
    # -------------------------------------------------------------
    # A) Inicialização da Câmera
    camera = Camera(
        calibration_file="config/camera_calibration.yaml",
        width=640,
        height=480,
        index=0
    )

    # B) Inicialização do modelo YOLOv8 (segmentação)
    vision = VisionSystem(
        model_path="models/best.pt",  # Ajuste para o seu modelo
        conf_threshold=0.5
    )

    # C) Perception: se você tiver camera_matrix/dist_coeffs já carregados,
    #    pode passá-los diretamente, ou deixar None se não for usar.
    #    Supondo que a altura da câmera é ~30 cm, alpha=0.98 para Filtro Complementar.
    perception = Perception(
        camera_matrix=None,
        dist_coeffs=None,
        camera_height=0.30,
        alpha=0.98
    )

    # D) Definição do campo (field_map)
    #    Ajuste conforme as dimensões do seu campo real.
    field_map = load_config("data/maps/field_map.json")

    # E) Inicialização do SLAM (Monte Carlo Localization)
    slam = MonteCarloLocalization(
        field_map=field_map,
        num_particles=300,
        x_var=0.1,
        y_var=0.1,
        theta_var=0.1
    )

    last_time = time.time()

    # -------------------------------------------------------------
    # 2) LOOP PRINCIPAL
    # -------------------------------------------------------------
    try:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now

            # 2.1) Captura frame da câmera
            frame = camera.get_frame()
            if frame is None:
                print("[Main] Não foi possível capturar frame. Encerrando...")
                break

            # 2.2) Leitura fictícia do IMU (substitua por leitura real no seu robô)
            #     Exemplo: cada loop, atualiza giroscópio e magnetômetro de forma aleatória ou fixa
            imu_data = {
                'timestamp': now,
                'accel': (0.0, 0.0, 9.81),
                'gyro': (0.0, 0.0, 0.0),   # rad/s
                'mag': (0.5, 0.2, 0.0)    # ex.: heading aproximado
            }
            perception.update_imu_data(imu_data)

            # 2.3) Detecção de landmarks via YOLOv8
            detections = vision.detect_landmarks(frame)

            # (Opcional) Desenhar no frame para debug
            debug_frame = vision.draw_segmentations(frame.copy(), detections)

            # 2.4) Converter as detecções para coordenadas do campo (x, y)
            #     Perception faz a fusão com IMU e retorna dicionário de landmarks
            world_positions = perception.process_detections(detections)
            #   Exemplo: {
            #     'ball': [(xb1, yb1), ...],
            #     'centercircle': [(xc, yc), ...],
            #     'goal': [(xg, yg), ...],
            #     ...
            #   }

            # 2.5) Cálculo de odometria (dx, dy, dtheta)
            #     Se você tiver algo real, extraia do IMU ou encoders. Aqui, colocamos algo fixo
            dx = 0.01
            dy = 0.0
            dtheta = 0.0
            odom = (dx, dy, dtheta)

            # 2.6) Transformar as posições do 'goal', 'centercircle', 'penaltycross' em medições (dist, angle)
            #     Para cada landmark que quiser usar no SLAM, a ideia é:
            #     dist = sqrt(x^2 + y^2)
            #     angle = atan2(y, x)
            slam_world_positions = {}

            # -- Goal
            if 'goal' in world_positions and len(world_positions['goal']) > 0:
                xg, yg = world_positions['goal'][0]  # pega a primeira detecção
                dist_g = math.sqrt(xg**2 + yg**2)
                angle_g = math.atan2(yg, xg)
                slam_world_positions['goal_measure'] = (dist_g, angle_g)

            # -- Center circle
            if 'centercircle' in world_positions and len(world_positions['centercircle']) > 0:
                xc, yc = world_positions['centercircle'][0]
                dist_c = math.sqrt(xc**2 + yc**2)
                angle_c = math.atan2(yc, xc)
                slam_world_positions['centercircle_measure'] = (dist_c, angle_c)

            # -- Penalty cross
            if 'penaltycross' in world_positions and len(world_positions['penaltycross']) > 0:
                xp, yp = world_positions['penaltycross'][0]
                dist_p = math.sqrt(xp**2 + yp**2)
                angle_p = math.atan2(yp, xp)
                slam_world_positions['penaltycross_measure'] = (dist_p, angle_p)

            # 2.7) Atualiza SLAM
            slam.update(odom, dt, slam_world_positions)
            # Pega a pose estimada
            x_est, y_est, theta_est = slam.get_best_estimate()

            # 2.8) Exibir no frame (para debug) a pose
            cv2.putText(debug_frame,
                        f"Pose: x={x_est:.2f}, y={y_est:.2f}, th={math.degrees(theta_est):.1f} deg",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

            # Mostra o frame de debug
            cv2.imshow("Debug Vision + SLAM", debug_frame)

            # Pressione 'q' para sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)  # Ajustar conforme desempenho desejado

    except KeyboardInterrupt:
        print("[Main] Encerrando por KeyboardInterrupt...")

    # Libera recursos
    camera.release()
    cv2.destroyAllWindows()
    print("[Main] Fim do programa.")


if __name__ == "__main__":
    main()
