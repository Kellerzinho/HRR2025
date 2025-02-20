import cv2
import numpy as np
import math

class FieldDrawer:
    def __init__(self, scale=100):
        """
        :param scale: Fator de escala (pixels por metro).
                      Ex.: scale=100 → 1 m = 100 px.
        """
        self.scale = scale

    def overlay_image(self, background, overlay, x, y):
        bg_height, bg_width = background.shape[:2]
        h, w = overlay.shape[:2]
        
        # Ajustar coordenadas se ultrapassarem os limites
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + w, bg_width)
        y2 = min(y + h, bg_height)
        
        if x2 <= x1 or y2 <= y1:
            return background
        
        # Recortar a parte visível do overlay
        overlay_cropped = overlay[y1 - y:y2 - y, x1 - x:x2 - x]
        
        if overlay_cropped.shape[2] == 4:
            alpha = overlay_cropped[:, :, 3:] / 255.0
            overlay_rgb = overlay_cropped[:, :, :3]
        else:
            alpha = np.ones_like(overlay_cropped[:, :, :1])
            overlay_rgb = overlay_cropped
        
        # Região de interesse no background
        bg_roi = background[y1:y2, x1:x2]
        
        # Combinar as imagens (alpha blending)
        bg_roi[:] = (bg_roi * (1 - alpha) + overlay_rgb * alpha).astype(np.uint8)
        
        return background

    # Método de conversão com depuração
    def world_to_field_coords(self, X, Y, field_draw_width, field_draw_height, field_map):
        """
        Converte coordenadas do mundo (em metros) para coordenadas em pixels,
        considerando o tamanho real do campo (field_map['width'] e field_map['height']).
        Adiciona mensagens de depuração para verificar os valores.
        """
        x_min = 0.0
        x_max = field_map["width"]   # Ex: 5.5
        y_min = 0.0
        y_max = field_map["height"]  # Ex: 3.5

        # Normaliza os valores para [0, 1]
        x_norm = (X - x_min) / (x_max - x_min)
        y_norm = (Y - y_min) / (y_max - y_min)

        # Mapeia para o espaço de desenho (0..field_draw_width, 0..field_draw_height)
        x_draw = int(x_norm * field_draw_width)
        # Inverte o eixo Y para que 0 fique no topo
        y_draw = int((1 - y_norm) * field_draw_height)

        # Depuração: imprimir coordenadas
        print(f"[DEBUG] world_to_field_coords: (X, Y)=({X:.3f}, {Y:.3f}) -> (x_norm, y_norm)=({x_norm:.3f}, {y_norm:.3f}) -> (x_draw, y_draw)=({x_draw}, {y_draw})")
        return x_draw, y_draw

    def draw_field(self, field_map, robot_pose, extra_landmarks=None):
        """
        Desenha o campo de 5.5m (largura) x 3.5m (altura) e insere:
         - Retângulo de contorno
         - Áreas de goleiro (exemplo)
         - Linha central
         - Círculo central
         - Gols
         - Marcas de pênalti
         - Posição do robô (triângulo)
         - Landmarks extras (bola, robôs etc.)
        """
        width_m = field_map["width"]   # 5.5 m
        height_m = field_map["height"] # 3.5 m
        field_width = int(width_m * self.scale)
        field_height = int(height_m * self.scale)
        
        # Criar imagem do campo (verde)
        field_img = np.zeros((field_height, field_width, 3), dtype=np.uint8)
        field_img[:] = (0, 128, 0)  # verde
        
        # Desenhar bordas do campo
        cv2.rectangle(field_img, (0, 0), (field_width-1, field_height-1), (255, 255, 255), 2)
        
        # Áreas do goleiro (exemplo)
        left_area_pts = [(0, 1.0), (1.0, 2.5)]
        p1 = (int(left_area_pts[0][0] * self.scale), 
              int(field_height - left_area_pts[0][1] * self.scale))
        p2 = (int(left_area_pts[1][0] * self.scale), 
              int(field_height - left_area_pts[1][1] * self.scale))
        cv2.rectangle(field_img, p1, p2, (255, 255, 255), 2)
        
        right_area_pts = [(4.5, 1.0), (5.5, 2.5)]
        p1 = (int(right_area_pts[0][0] * self.scale), 
              int(field_height - right_area_pts[0][1] * self.scale))
        p2 = (int(right_area_pts[1][0] * self.scale), 
              int(field_height - right_area_pts[1][1] * self.scale))
        cv2.rectangle(field_img, p1, p2, (255, 255, 255), 2)
        
        # Linha central
        cv2.line(field_img,
                 (field_width // 2, 0),
                 (field_width // 2, field_height),
                 (255, 255, 255), 2)
        
        # Círculo central
        center_x = field_map["center_circle_x"] * self.scale  # 2.75 * scale
        center_y = field_height - field_map["center_circle_y"] * self.scale
        radius_m = 0.75  # Raio em metros
        radius_px = int(radius_m * self.scale)
        cv2.circle(field_img, (int(center_x), int(center_y)), radius_px, (255, 255, 255), 2)
        
        # Desenhar símbolo (opcional)
        ime_logo = cv2.imread('data/ime_logo.png', cv2.IMREAD_UNCHANGED)
        if ime_logo is not None:
            logo_size = int(radius_px * 1.2)
            ime_logo = cv2.resize(ime_logo, (logo_size, logo_size))
            x_logo = int(center_x - logo_size // 2)
            y_logo = int(center_y - logo_size // 2)
            field_img = self.overlay_image(field_img, ime_logo, x_logo, y_logo)
        
        # Gols (círculos)
        lg_x = field_map["left_goal_x"]
        lg_y = field_map["left_goal_y"]
        lg_px, lg_py = self.world_to_field_coords(lg_x, lg_y, field_width, field_height, field_map)
        cv2.circle(field_img, (lg_px, lg_py), 10, (0, 0, 255), -1)
        
        rg_x = field_map["right_goal_x"]
        rg_y = field_map["right_goal_y"]
        rg_px, rg_py = self.world_to_field_coords(rg_x, rg_y, field_width, field_height, field_map)
        cv2.circle(field_img, (rg_px, rg_py), 10, (255, 0, 0), -1)
        
        # Cruzes de pênalti
        lpc_x = field_map["left_penaltycross_x"]
        lpc_y = field_map["left_penaltycross_y"]
        lpc_px, lpc_py = self.world_to_field_coords(lpc_x, lpc_y, field_width, field_height, field_map)
        cv2.circle(field_img, (lpc_px, lpc_py), 5, (255, 255, 255), -1)
        
        rpc_x = field_map["right_penaltycross_x"]
        rpc_y = field_map["right_penaltycross_y"]
        rpc_px, rpc_py = self.world_to_field_coords(rpc_x, rpc_y, field_width, field_height, field_map)
        cv2.circle(field_img, (rpc_px, rpc_py), 5, (255, 255, 255), -1)
        
        # Desenhar robô (triângulo)
        x_robot, y_robot, theta = robot_pose
        robot_px, robot_py = self.world_to_field_coords(x_robot, y_robot, field_width, field_height, field_map)
        size = 15  # tamanho do triângulo em pixels

        front_x = robot_px + int(size * math.cos(theta))
        front_y = robot_py - int(size * math.sin(theta))
        left_x = robot_px + int((size/2) * math.cos(theta + 2.5))
        left_y = robot_py - int((size/2) * math.sin(theta + 2.5))
        right_x = robot_px + int((size/2) * math.cos(theta - 2.5))
        right_y = robot_py - int((size/2) * math.sin(theta - 2.5))

        pts = np.array([[front_x, front_y],
                        [left_x,  left_y],
                        [right_x, right_y]], dtype=np.int32)
        cv2.fillConvexPoly(field_img, pts, (0, 255, 255))
        print(f"[DEBUG] Robot: world=({x_robot:.3f}, {y_robot:.3f}), drawn=({robot_px}, {robot_py}), theta={theta:.2f}")

        # Desenhar landmarks extras (bola, robôs, etc.)
        if extra_landmarks is not None:
            # Desenhar bola
            if "ball" in extra_landmarks:
                print("[DEBUG] Desenhando", len(extra_landmarks["ball"]), "bola(s)")
                for (bx, by) in extra_landmarks["ball"]:
                    bx_px, by_px = self.world_to_field_coords(bx, by, field_width, field_height, field_map)
                    cv2.circle(field_img, (bx_px, by_px), 6, (0, 0, 255), -1)
                    print(f"[DEBUG] Bola: world=({bx:.3f}, {by:.3f}) -> drawn=({bx_px}, {by_px})")
                    
                    if "ball_velocity" in extra_landmarks:
                        vx, vy = extra_landmarks["ball_velocity"]
                        scale_v = 50  # pixels por m/s
                        end_x = int(bx_px + vx * scale_v)
                        end_y = int(by_px - vy * scale_v)
                        cv2.arrowedLine(field_img, (bx_px, by_px), (end_x, end_y), (0, 255, 0), 2, tipLength=0.3)
                        print(f"[DEBUG] Velocidade bola: vx={vx:.2f}, vy={vy:.2f} -> arrow end=({end_x}, {end_y})")
            else:
                print("[DEBUG] Nenhuma bola recebida para desenhar")
            
            # Desenhar robôs extras
            if "robot" in extra_landmarks:
                for (rx, ry) in extra_landmarks["robot"]:
                    rx_px, ry_px = self.world_to_field_coords(rx, ry, field_width, field_height, field_map)
                    cv2.circle(field_img, (rx_px, ry_px), 8, (255, 0, 255), -1)
                    print(f"[DEBUG] Robô extra: world=({rx:.3f}, {ry:.3f}) -> drawn=({rx_px}, {ry_px})")
        
        return field_img
