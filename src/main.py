import time
from core.camera import Camera
from core.vision import VisionSystem
from core.perception import Perception
from core.slam import MonteCarloLocalization
from core.strategy import Strategy
from core.navigation import Navigator
from core.control_pid import MotionController
from utils.config_loader import load_config

def main():
    # Carrega configs
    roles_config = load_config("config/roles.yaml")
    camera_conf = load_config("config/camera_calibration.yaml")
    field_map = load_config("data/maps/field_map.json")  # ex.: dimensões do campo

    # Instancia módulos
    camera = Camera(calibration_file="config/camera_calibration.yaml")
    vision = VisionSystem("models/best.pt")
    perception = Perception(camera_conf)
    slam = MonteCarloLocalization(field_map)
    strategy = Strategy(role="atacante")  # foco no robô artilheiro
    navigator = Navigator()
    motion = MotionController()

    # Inicializa variáveis de estado
    robot_pose = (0,0,0) # x,y,theta
    last_time = time.time()

    try:
        while True:
            start_loop = time.time()

            # 1) Captura frame e processa
            frame = camera.get_frame()
            if frame is None:
                continue

            detections = vision.detect_objects(frame)
            field_mask = vision.segment_field(frame)  # se precisar

            # 2) Atualiza percepção com dados de IMU (supõe que IMU é lido em outro lugar e enviado aqui)
            # Exemplo fictício de IMU:
            # imu_data = {'timestamp': start_loop, 'accel': (..), 'gyro': (..), 'mag': (..)}
            # perception.update_imu_data(imu_data)
            orientation = perception.fuse(frame_timestamp=start_loop)

            # 3) Converte detecções para coordenadas do mundo
            world_detections = []
            for d in detections:
                wpos = perception.compute_world_coords(d, orientation)
                world_detections.append((d['label'], wpos))

            # 4) SLAM / Localização
            dt = start_loop - last_time
            last_time = start_loop
            # ex.: odometria = (dx, dy, dtheta) -> derivado de IMU, passadas, etc.
            odom = (0.01, 0.0, 0.0)  # fictício
            slam.update(world_detections, odom, dt)
            robot_pose = slam.get_best_estimate()

            # 5) Monta o estado do mundo para a estratégia
            world_state = {
                'robot_pose': robot_pose,
                'ball_position': None,
                'goal_position': (field_map['goal_x'], field_map['goal_y']),
                # Se YOLO detectou "bola" nas world_detections, setar:
            }
            for label, wpos in world_detections:
                if label == "bola":
                    world_state['ball_position'] = (wpos[0], wpos[1])

            # 6) Estratégia: decide ação
            action, target = strategy.decide(world_state)

            # 7) Navegação + Controle
            if action == "move_to" and target is not None:
                path = navigator.plan_path(robot_pose, target)
                if path:
                    next_wp = path[0]
                    v_lin, v_ang = navigator.compute_velocity(robot_pose, next_wp)
                    linear_out, angular_out = motion.step(v_lin, v_ang, dt, {})
                    # Enviar comandos para servos (abstraído)
            elif action == "search":
                # Rodar em círculo ou mexer a cabeça
                pass
            else:
                # idle
                pass

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Encerrando programa...")
    finally:
        camera.release()


if __name__ == "__main__":
    main()
