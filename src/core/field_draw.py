import cv2
import numpy as np
import math

class FieldDrawer:
    def __init__(self, scale=100):
        self.scale = scale

    def overlay_image(self, background, overlay, x, y):
        bg_height, bg_width = background.shape[:2]
        h, w = overlay.shape[0], overlay.shape[1]
        
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
        
        # Combinar as imagens
        bg_roi[:] = (bg_roi * (1 - alpha) + overlay_rgb * alpha).astype(np.uint8)
        
        return background

    def draw_field(self, field_map, robot_pose, extra_landmarks=None):
        width_m = field_map["width"]
        height_m = field_map["height"]
        field_width = int(width_m * self.scale)
        field_height = int(height_m * self.scale)
        
        # Criar imagem do campo (verde)
        field_img = np.zeros((field_height, field_width, 3), dtype=np.uint8)
        field_img[:] = (0, 128, 0)
        
        # Desenhar bordas do campo
        cv2.rectangle(field_img, (0, 0), (field_width-1, field_height-1), (255, 255, 255), 2)
        
        # Áreas dos goleiros (retângulos)
        # Área esquerda (x: 0-1.5m, y: 2-4m)
        left_area_pts = [
            (0, 0.5),
            (1, 3)
        ]
        px1 = (int(left_area_pts[0][0] * self.scale), int(field_height - left_area_pts[0][1] * self.scale))
        px2 = (int(left_area_pts[1][0] * self.scale), int(field_height - left_area_pts[1][1] * self.scale))
        cv2.rectangle(field_img, px1, px2, (255, 255, 255), 2)
        
        # Área direita (x: 7.5-9m, y: 2-4m)
        right_area_pts = [
            (4.5, 0.5),
            (5.5, 3)
        ]
        px1 = (int(right_area_pts[0][0] * self.scale), int(field_height - right_area_pts[0][1] * self.scale))
        px2 = (int(right_area_pts[1][0] * self.scale), int(field_height - right_area_pts[1][1] * self.scale))
        cv2.rectangle(field_img, px1, px2, (255, 255, 255), 2)
        
        # Linha central
        cv2.line(field_img, (field_width//2, 0), (field_width//2, field_height), (255, 255, 255), 2)
        
        # Círculo central (raio 0.75m para maior visibilidade)
        center_x = field_map["center_circle_x"] * self.scale
        center_y = field_height - field_map["center_circle_y"] * self.scale
        radius = int(0.5 * self.scale)
        cv2.circle(field_img, (int(center_x), int(center_y)), radius, (255, 255, 255), 2)
        
        # Desenhar símbolo do IME
        ime_logo = cv2.imread('data/ime_logo.png', cv2.IMREAD_UNCHANGED)
        if ime_logo is not None:
            logo_size = int(1.2 * radius)  # Tamanho ajustado
            ime_logo = cv2.resize(ime_logo, (logo_size, logo_size))
            x = int(center_x - logo_size // 2)
            y = int(center_y - logo_size // 2)
            field_img = self.overlay_image(field_img, ime_logo, x, y)
        
        # Gols (círculos)
        # Gol esquerdo (vermelho)
        lg_x = int(field_map["left_goal_x"] * self.scale)
        lg_y = int(field_height - field_map["left_goal_y"] * self.scale)
        cv2.circle(field_img, (lg_x, lg_y), 10, (0, 0, 255), -1)
        
        # Gol direito (azul)
        rg_x = int(field_map["right_goal_x"] * self.scale)
        rg_y = int(field_height - field_map["right_goal_y"] * self.scale)
        cv2.circle(field_img, (rg_x, rg_y), 10, (255, 0, 0), -1)
        
        # Cruzes de pênalti
        # Esquerda
        lpc_x = int(field_map["left_penaltycross_x"] * self.scale)
        lpc_y = int(field_height - field_map["left_penaltycross_y"] * self.scale)
        cv2.circle(field_img, (lpc_x, lpc_y), 5, (255, 255, 255), -1)
        
        # Direita
        rpc_x = int(field_map["right_penaltycross_x"] * self.scale)
        rpc_y = int(field_height - field_map["right_penaltycross_y"] * self.scale)
        cv2.circle(field_img, (rpc_x, rpc_y), 5, (255, 255, 255), -1)
        
        # Desenhar robô (triângulo)
        x, y, theta = robot_pose
        robot_px = int(x * self.scale)
        robot_py = int(field_height - y * self.scale)
        size = 15
        pts = np.array([
            [robot_px + int(size * math.cos(theta)), robot_py - int(size * math.sin(theta))],
            [robot_px + int(size/2 * math.cos(theta + 2.5)), robot_py - int(size/2 * math.sin(theta + 2.5))],
            [robot_px + int(size/2 * math.cos(theta - 2.5)), robot_py - int(size/2 * math.sin(theta - 2.5))]
        ])
        cv2.fillConvexPoly(field_img, pts, (0, 255, 255))

        # Desenhar landmarks extras, se houver
        if extra_landmarks is not None:
            # Desenhar a bola como um círculo vermelho
            if "ball" in extra_landmarks:
                for (bx, by) in extra_landmarks["ball"]:
                    bx_px = int(bx * self.scale)
                    by_px = int(field_height - by * self.scale)
                    cv2.circle(field_img, (bx_px, by_px), 6, (0, 0, 255), -1)
            
            # Exemplo para linhas: desenha como pequenos pontos verdes
            #if "line" in extra_landmarks:
                #for (lx, ly) in extra_landmarks["line"]:
                #    lx_px = int(lx * self.scale)
                #    ly_px = int(field_height - ly * self.scale)
                #    cv2.circle(field_img, (lx_px, ly_px), 4, (0, 255, 0), -1)
            
            # Exemplo para outros robôs: desenha como círculos magenta
            if "robot" in extra_landmarks:
                for (rx, ry) in extra_landmarks["robot"]:
                    rx_px = int(rx * self.scale)
                    ry_px = int(field_height - ry * self.scale)
                    cv2.circle(field_img, (rx_px, ry_px), 8, (255, 0, 255), -1)
        
        
        return field_img