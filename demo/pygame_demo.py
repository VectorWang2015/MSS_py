import math
from dataclasses import dataclass

import numpy as np

from MSS_simulator_py.osv.model import OSVDynamics, OSVEnvironment
from MSS_simulator_py.osv.params import load_osv_custom_params

from .demo_controls import DemoConfig, DemoControlState, apply_action


@dataclass
class DemoRuntimeConfig:
    width: int = 1360
    height: int = 820
    dt: float = 0.02
    map_scale_px_per_m: float = 2.0
    body_scale_px_per_m: float = 3.0
    force_scale_px_per_n: float = 1.0 / 8000.0
    max_path_points: int = 3000
    left_ratio: float = 0.62


class PygameOSVDemo:
    def __init__(self, runtime_cfg: DemoRuntimeConfig | None = None):
        self.cfg = runtime_cfg if runtime_cfg is not None else DemoRuntimeConfig()
        self.model = OSVDynamics(params=load_osv_custom_params())
        self.control_cfg = DemoConfig(rpm_limits=self.model.params.n_max.copy())
        self.control_state = DemoControlState()
        self.state = np.zeros(12)
        self.path_world: list[tuple[float, float]] = []

    def _key_to_action(self, key: int) -> str | None:
        import pygame

        mapping = {
            pygame.K_q: "n1_up",
            pygame.K_a: "n1_down",
            pygame.K_w: "n2_up",
            pygame.K_s: "n2_down",
            pygame.K_e: "n3_up",
            pygame.K_d: "n3_down",
            pygame.K_r: "n4_up",
            pygame.K_f: "n4_down",
            pygame.K_z: "a1_left",
            pygame.K_x: "a1_right",
            pygame.K_c: "a2_left",
            pygame.K_v: "a2_right",
            pygame.K_t: "current_speed_up",
            pygame.K_g: "current_speed_down",
            pygame.K_y: "current_dir_left",
            pygame.K_h: "current_dir_right",
            pygame.K_i: "tau_x_plus",
            pygame.K_k: "tau_x_minus",
            pygame.K_o: "tau_y_plus",
            pygame.K_l: "tau_y_minus",
            pygame.K_p: "tau_n_plus",
            pygame.K_SEMICOLON: "tau_n_minus",
            pygame.K_SPACE: "toggle_pause",
            pygame.K_BACKSPACE: "zero_controls",
        }
        return mapping.get(key)

    def _reset_state(self):
        self.state = np.zeros(12)
        self.path_world.clear()

    def _panel_rects(self):
        left_w = int(self.cfg.width * self.cfg.left_ratio)
        right_w = self.cfg.width - left_w
        return (0, 0, left_w, self.cfg.height), (left_w, 0, right_w, self.cfg.height)

    def _left_center(self):
        left, _ = self._panel_rects()
        return left[0] + left[2] / 2.0, left[1] + left[3] / 2.0

    def _map_world_to_screen(
        self, north: float, east: float, ref_n: float, ref_e: float
    ):
        cx, cy = self._left_center()
        x = cx + (east - ref_e) * self.cfg.map_scale_px_per_m
        y = cy - (north - ref_n) * self.cfg.map_scale_px_per_m
        return float(x), float(y)

    def _body_to_right_screen(self, x_b: float, y_b: float):
        _, right = self._panel_rects()
        cx = right[0] + right[2] / 2.0
        cy = right[1] + right[3] / 2.0
        sx = cx + y_b * self.cfg.body_scale_px_per_m
        sy = cy - x_b * self.cfg.body_scale_px_per_m
        return float(sx), float(sy)

    def _compute_left_ship_triangle(self, psi: float):
        cx, cy = self._left_center()
        length = 48.0
        half_width = 16.0

        f_n = math.cos(psi)
        f_e = math.sin(psi)
        s_n = -math.sin(psi)
        s_e = math.cos(psi)

        f_dx = f_e * self.cfg.map_scale_px_per_m
        f_dy = -f_n * self.cfg.map_scale_px_per_m
        s_dx = s_e * self.cfg.map_scale_px_per_m
        s_dy = -s_n * self.cfg.map_scale_px_per_m

        nose = (cx + length * f_dx, cy + length * f_dy)
        stern = (cx - 0.55 * length * f_dx, cy - 0.55 * length * f_dy)
        port = (stern[0] - half_width * s_dx, stern[1] - half_width * s_dy)
        starboard = (stern[0] + half_width * s_dx, stern[1] + half_width * s_dy)
        return nose, port, starboard

    def _compute_thruster_vectors_body(self, control: np.ndarray):
        p = self.model.params
        rpm = np.asarray(control[:4], dtype=float)
        alpha = np.asarray(control[4:6], dtype=float)
        thrust = np.diag(p.k_thr) * np.abs(rpm) * rpm

        vectors = []
        for i in range(4):
            x_b = p.l_x[i]
            y_b = p.l_y[i]
            if i < 2:
                fxb = 0.0
                fyb = thrust[i]
            else:
                az = alpha[i - 2]
                fxb = thrust[i] * math.cos(az)
                fyb = thrust[i] * math.sin(az)
            vectors.append((x_b, y_b, fxb, fyb, i))
        return vectors

    def _draw_text(
        self, surface, font, x: int, y: int, text: str, color=(220, 230, 235)
    ):
        label = font.render(text, True, color)
        surface.blit(label, (x, y))

    def _draw_left_panel(self, surface):
        import pygame

        left, _ = self._panel_rects()
        panel = pygame.Rect(*left)
        pygame.draw.rect(surface, (16, 26, 36), panel)

        north = float(self.state[6])
        east = float(self.state[7])
        psi = float(self.state[11])

        for i in range(-12, 13):
            x0, y0 = self._map_world_to_screen(north - 300, east + i * 20, north, east)
            x1, y1 = self._map_world_to_screen(north + 300, east + i * 20, north, east)
            pygame.draw.line(surface, (30, 42, 54), (x0, y0), (x1, y1), 1)
            x2, y2 = self._map_world_to_screen(north + i * 20, east - 300, north, east)
            x3, y3 = self._map_world_to_screen(north + i * 20, east + 300, north, east)
            pygame.draw.line(surface, (30, 42, 54), (x2, y2), (x3, y3), 1)

        if len(self.path_world) > 1:
            path = [
                self._map_world_to_screen(n, e, north, east) for n, e in self.path_world
            ]
            pygame.draw.lines(surface, (84, 170, 255), False, path, 2)

        nose, port, starboard = self._compute_left_ship_triangle(psi)
        pygame.draw.polygon(surface, (236, 224, 145), [nose, port, starboard])
        pygame.draw.circle(
            surface,
            (255, 255, 255),
            (int(self._left_center()[0]), int(self._left_center()[1])),
            2,
        )

        title_font = pygame.font.SysFont("monospace", 18)
        small = pygame.font.SysFont("monospace", 15)
        self._draw_text(
            surface, title_font, left[0] + 16, 16, "Map View (ship centered)"
        )
        self._draw_text(
            surface,
            small,
            left[0] + 16,
            42,
            f"NED: N={north:7.2f} m  E={east:7.2f} m  psi={math.degrees(psi):6.2f} deg",
        )

    def _draw_right_panel(self, surface):
        import pygame

        _, right = self._panel_rects()
        panel = pygame.Rect(*right)
        pygame.draw.rect(surface, (22, 20, 26), panel)

        cx, cy = self._body_to_right_screen(0.0, 0.0)
        p = self.model.params

        corners_body = [
            (p.l / 2.0, -p.b / 2.0),
            (p.l / 2.0, p.b / 2.0),
            (-p.l / 2.0, p.b / 2.0),
            (-p.l / 2.0, -p.b / 2.0),
        ]
        hull = [self._body_to_right_screen(xb, yb) for xb, yb in corners_body]
        pygame.draw.polygon(surface, (200, 200, 210), hull, width=2)

        u = self.control_state.to_control_vector(self.control_cfg)
        vectors = self._compute_thruster_vectors_body(u)
        colors = [(255, 160, 90), (255, 210, 100), (120, 255, 170), (120, 220, 255)]
        labels = ["n1", "n2", "n3", "n4"]
        font = pygame.font.SysFont("monospace", 15)

        for xb, yb, fxb, fyb, idx in vectors:
            sx, sy = self._body_to_right_screen(xb, yb)
            dx = fyb * self.cfg.force_scale_px_per_n
            dy = -fxb * self.cfg.force_scale_px_per_n
            length = math.hypot(dx, dy)
            if length > 0.0 and length < 12.0:
                scale = 12.0 / length
                dx *= scale
                dy *= scale
            ex = sx + dx
            ey = sy + dy
            pygame.draw.circle(surface, colors[idx], (int(sx), int(sy)), 4)
            pygame.draw.line(surface, colors[idx], (sx, sy), (ex, ey), 3)
            self._draw_text(
                surface, font, int(sx + 6), int(sy + 2), labels[idx], colors[idx]
            )

        title_font = pygame.font.SysFont("monospace", 18)
        small = pygame.font.SysFont("monospace", 14)
        self._draw_text(surface, title_font, right[0] + 16, 16, "Body / Actuator View")
        u_b, v_b, w_b, p_b, q_b, r_b = self.state[:6]
        self._draw_text(
            surface,
            small,
            right[0] + 16,
            44,
            f"BODY nu: u={u_b:6.3f} v={v_b:6.3f} w={w_b:6.3f} p={p_b:6.3f} q={q_b:6.3f} r={r_b:6.3f}",
        )

        rpm = self.control_state.rpm
        az = self.control_state.azimuth
        self._draw_text(
            surface,
            small,
            right[0] + 16,
            68,
            f"rpm: n1={rpm[0]:6.1f} n2={rpm[1]:6.1f} n3={rpm[2]:6.1f} n4={rpm[3]:6.1f}",
        )
        self._draw_text(
            surface,
            small,
            right[0] + 16,
            90,
            f"a1={math.degrees(az[0]):6.1f} deg  a2={math.degrees(az[1]):6.1f} deg  (starboard/right is +)",
        )
        self._draw_text(
            surface,
            small,
            right[0] + 16,
            112,
            "n1,n2: bow tunnel; n3,n4: stern azimuth (demo custom: n3=n4)",
        )

        vc = self.control_state.current_speed
        beta = self.control_state.current_direction
        psi = float(self.state[11])
        u_c = vc * math.cos(beta - psi)
        v_c = vc * math.sin(beta - psi)
        env = OSVEnvironment(
            current_speed=vc,
            current_direction=beta,
            tau_env_ned=self.control_state.tau_env_ned,
        )
        tau_body = self.model._tau_env_body(self.state[6:12], env)
        self._draw_text(
            surface,
            small,
            right[0] + 16,
            136,
            f"BODY current: u_c={u_c:6.3f} v_c={v_c:6.3f} m/s",
        )
        self._draw_text(
            surface,
            small,
            right[0] + 16,
            158,
            f"BODY tau_env: X={tau_body[0]:7.1f} Y={tau_body[1]:7.1f} N={tau_body[5]:7.1f}",
        )

        self._draw_text(
            surface,
            small,
            right[0] + 16,
            self.cfg.height - 110,
            "q/a,w/s: n1,n2 bow tunnel +/-",
        )
        self._draw_text(
            surface,
            small,
            right[0] + 16,
            self.cfg.height - 88,
            "e/d,r/f: n3,n4 stern azimuth +/-",
        )
        self._draw_text(
            surface,
            small,
            right[0] + 16,
            self.cfg.height - 66,
            "z/x: a1 +/-   c/v: a2 +/-",
        )
        self._draw_text(
            surface,
            small,
            right[0] + 16,
            self.cfg.height - 44,
            "t/g: Vc +/-   y/h: beta(NED) +/-",
        )
        self._draw_text(
            surface,
            small,
            right[0] + 16,
            self.cfg.height - 22,
            "i/k,o/l,p/;: tau_NED_X/Y/N +/-  space: pause",
        )

        pygame.draw.line(
            surface, (55, 55, 70), (right[0], 0), (right[0], self.cfg.height), 2
        )

    def _draw_ui(self, surface):
        self._draw_left_panel(surface)
        self._draw_right_panel(surface)

    def run(self):
        import pygame

        pygame.init()
        pygame.display.set_caption("OSV Interactive Demo")
        screen = pygame.display.set_mode((self.cfg.width, self.cfg.height))
        clock = pygame.time.Clock()
        accumulator = 0.0
        running = True

        while running:
            frame_dt = clock.tick(60) / 1000.0
            accumulator += frame_dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self._reset_state()
                    action = self._key_to_action(event.key)
                    if action is not None:
                        self.control_state = apply_action(
                            self.control_state, self.control_cfg, action
                        )

            while accumulator >= self.cfg.dt:
                accumulator -= self.cfg.dt
                if not self.control_state.paused:
                    u = self.control_state.to_control_vector(self.control_cfg)
                    env = OSVEnvironment(
                        current_speed=self.control_state.current_speed,
                        current_direction=self.control_state.current_direction,
                        tau_env_ned=self.control_state.tau_env_ned,
                    )
                    self.state = self.model.step_rk4(self.state, u, self.cfg.dt, env)
                    self.path_world.append((float(self.state[6]), float(self.state[7])))
                    if len(self.path_world) > self.cfg.max_path_points:
                        self.path_world = self.path_world[-self.cfg.max_path_points :]

            self._draw_ui(screen)
            pygame.display.flip()

        pygame.quit()


def run_pygame_demo():
    PygameOSVDemo().run()
