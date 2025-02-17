# src/main.py

import time
import math
import cv2

# Import dos módulos
from core.camera import Camera
from core.vision import VisionSystem
from core.perception import Perception
from core.slam import MonteCarloLocalization
from core.imu import IMUReader
from core.odometry import OdometryState
from utils.config_loader import load_config

def main():
    """
    Loop principal que:
      - Captura frame da câmera
      - Detecta landmarks (YOLOv8)
      - Converte para coordenadas do campo (perception)
      - Integra aceleração do IMU (odometry)
      - Alimenta MCL com (dx, dy, dtheta) e medições de landmarks
    """

    # 1) Inicialização
    camera = Camera("config/camera_calibration.yaml", width=640, height=480, index=0)
    vision = VisionSystem(model_path="models/best.pt", conf_threshold=0.5)
    perception = Perception(camera_matrix=None, dist_coeffs=None, camera_height=0.30, alpha=0.98)

    # Carrega configurações do campo (com várias referências)
    field_map = load_config("data/maps/field_map.json")

    slam = MonteCarloLocalization(field_map=field_map, num_particles=300, x_var=0.1, y_var=0.1, theta_var=0.1)

    imu_reader = IMUReader(mpu_address=0x68, ak_address=0x0C, bus_id=1)
    odom_state = OdometryState()

    last_time = time.time()

    try:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now

            # 1) Frame da câmera
            frame = camera.get_frame()
            if frame is None:
                print("[Main] Falha ao capturar frame.")
                break

            # 2) Ler IMU e atualizar Perception
            imu_data = imu_reader.get_imu_data()
            perception.update_imu_data(imu_data)

            # 3) Detectar landmarks
            detections = vision.detect_landmarks(frame)
            debug_frame = vision.draw_segmentations(frame.copy(), detections)

            # 4) Converter detecções para (x, y) no campo
            world_positions_xy = perception.process_detections(detections)
            # ex.: {
            #   'left_goal': [(x,y), ...],
            #   'right_goal': [(x,y), ...],
            #   ...
            # }

            # 5) Odometria via aceleração do IMU
            dx, dy, dtheta = odom_state.compute_odometry_from_imu(perception)
            odom = (dx, dy, dtheta)

            # 6) Converter cada landmark detectado em (dist, angle)
            slam_world_positions = {}

            # Exemplo: se 'left_goal' foi detectado
            if 'left_goal' in world_positions_xy and len(world_positions_xy['left_goal']) > 0:
                lx, ly = world_positions_xy['left_goal'][0]
                dist_lg = math.sqrt(lx*lx + ly*ly)
                angle_lg = math.atan2(ly, lx)
                slam_world_positions['left_goal_measure'] = (dist_lg, angle_lg)

            if 'right_goal' in world_positions_xy and len(world_positions_xy['right_goal']) > 0:
                rx, ry = world_positions_xy['right_goal'][0]
                dist_rg = math.sqrt(rx*rx + ry*ry)
                angle_rg = math.atan2(ry, rx)
                slam_world_positions['right_goal_measure'] = (dist_rg, angle_rg)

            if 'center_circle' in world_positions_xy and len(world_positions_xy['center_circle']) > 0:
                cx, cy = world_positions_xy['center_circle'][0]
                dist_cc = math.sqrt(cx*cx + cy*cy)
                angle_cc = math.atan2(cy, cx)
                slam_world_positions['center_circle_measure'] = (dist_cc, angle_cc)

            if 'left_penaltycross' in world_positions_xy and len(world_positions_xy['left_penaltycross']) > 0:
                px, py = world_positions_xy['left_penaltycross'][0]
                dist_lp = math.sqrt(px*px + py*py)
                angle_lp = math.atan2(py, px)
                slam_world_positions['left_penaltycross_measure'] = (dist_lp, angle_lp)

            if 'right_penaltycross' in world_positions_xy and len(world_positions_xy['right_penaltycross']) > 0:
                px, py = world_positions_xy['right_penaltycross'][0]
                dist_rp = math.sqrt(px*px + py*py)
                angle_rp = math.atan2(py, px)
                slam_world_positions['right_penaltycross_measure'] = (dist_rp, angle_rp)

            # 7) Atualiza SLAM
            slam.update(odom, dt, slam_world_positions)
            x_est, y_est, theta_est = slam.get_best_estimate()

            # 8) Depuração
            cv2.putText(debug_frame,
                        f"Pose: x={x_est:.2f}, y={y_est:.2f}, th={math.degrees(theta_est):.1f}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            cv2.imshow("Debug Vision+SLAM", debug_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("[Main] Encerrando por KeyboardInterrupt...")

    camera.release()
    cv2.destroyAllWindows()
    print("[Main] Programa encerrado.")

if __name__ == "__main__":
    main()
