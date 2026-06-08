import os
import sys
import tkinter as tk
from tkinter import filedialog
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import numpy as np
import logging
import json

logger = logging.getLogger(__name__)

UI_PANEL_HEIGHT = 50

class MapEditor:
    def __init__(
            self,
            map_size: tuple[int, int] = (30, 30),
            window_size: tuple[int, int] = (800, 850)
    ):
        pygame.init()

        self.map_width = map_size[0]
        self.map_height = map_size[1]
        self.win_w, self.win_h = window_size

        self.city_map = np.zeros((self.map_height, self.map_width), dtype=np.uint8)

        self.cell_colors = {
            0: (200, 200, 200),
            1: (29, 95, 153),
            2: (85, 255, 255),
            3: (255, 241, 82),
            4: (255, 0, 0)
        }

        self.block_names = {
            0: "Road (0)",
            1: "Wall (1)",
            2: "Agent spawn (2)",
            3: "Bot spawn (3)",
            4: "Finish (4)"
        }

        self.current_block = 1
        self.drawing = False
        self.erasing = False
        self.clock = pygame.time.Clock()
        self.window = None

        self.pairing_mode = False
        self.pairs = []  # [{'spawn': (x,y), 'finish': (x,y)}, ...]
        self.selected_spawn = None  # (x, y)

        self._init_display()

    def _update_metrics(self):
        map_area_h = self.win_h - UI_PANEL_HEIGHT
        self.cell_w = self.win_w / self.map_width
        self.cell_h = map_area_h / self.map_height
        font_size = max(12, self.win_w // 42)
        self.font = pygame.font.SysFont(None, font_size)
        pair_font_size = max(12, int(min(self.cell_w, self.cell_h) * 0.8))
        self.pair_font = pygame.font.SysFont(None, pair_font_size)

    def _init_display(self):
        self._update_metrics()
        self.window = pygame.display.set_mode((self.win_w, self.win_h), pygame.RESIZABLE)
        pygame.display.set_caption("Environment Map Editor")

    def _save_map(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save the map"
        )
        root.destroy()
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for row in self.city_map:
                        f.write(' '.join(map(str, row)) + '\n')
                print(f"[INFO] Map saved: {file_path}")
            except Exception as e:
                print(f"[ERROR] Error while saving map: {e}")

    def _load_map(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Load the map"
        )
        root.destroy()
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    rows = f.readlines()
                new_map = []
                for row in rows:
                    if row.strip():
                        new_map.append([int(x) for x in row.strip().split()])
                self.city_map = np.array(new_map, dtype=np.uint8)
                self.map_height, self.map_width = self.city_map.shape
                self._update_metrics()
                print(f"[INFO] Map loaded: {file_path}")
            except Exception as e:
                print(f"[ERROR] Error while loading: {e}")

    def _resize_map(self):
        result = {}

        dialog = tk.Tk()
        dialog.title("Map size")
        dialog.resizable(False, False)
        dialog.geometry("200x120")
        dialog.eval('tk::PlaceWindow . center')

        font = ("Arial", 12)

        tk.Label(dialog, text="Width:", font=font).grid(row=0, column=0, padx=20, pady=14, sticky="e")
        entry_w = tk.Entry(dialog, width=6, font=font)
        entry_w.insert(0, str(self.map_width))
        entry_w.grid(row=0, column=1, padx=20, pady=14)

        tk.Label(dialog, text="Height:", font=font).grid(row=1, column=0, padx=20, pady=14, sticky="e")
        entry_h = tk.Entry(dialog, width=6, font=font)
        entry_h.insert(0, str(self.map_height))
        entry_h.grid(row=1, column=1, padx=20, pady=14)

        def confirm():
            try:
                result['w'] = int(entry_w.get())
                result['h'] = int(entry_h.get())
                dialog.destroy()
            except ValueError:
                pass

        tk.Button(dialog, text="OK", font=font, command=confirm).grid(row=2, column=0, columnspan=2, pady=8)
        dialog.bind("<Return>", lambda e: confirm())
        dialog.mainloop()

        w, h = result.get('w'), result.get('h')
        if w and h and w > 0 and h > 0:
            new_map = np.zeros((h, w), dtype=np.uint8)
            min_h = min(self.map_height, h)
            min_w = min(self.map_width, w)
            new_map[:min_h, :min_w] = self.city_map[:min_h, :min_w]
            self.city_map = new_map
            self.map_width = w
            self.map_height = h
            self._update_metrics()
            print(f"[INFO] Size changed to {w}x{h}")

    def _apply_brush(self, mouse_pos):
        x, y = mouse_pos
        grid_x = int(x / self.cell_w)
        grid_y = int(y / self.cell_h)
        if 0 <= grid_x < self.map_width and 0 <= grid_y < self.map_height:
            if self.drawing:
                self.city_map[grid_y, grid_x] = self.current_block
            elif self.erasing:
                self.city_map[grid_y, grid_x] = 0

    def _save_config(self):
        if not self.pairs:
            print("[WARN] No pairs to save")
            return
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save agent config"
        )
        root.destroy()
        if file_path:
            try:
                config = {"agents": [
                    {"spawn": list(p['spawn']), "finish": list(p['finish'])}
                    for p in self.pairs
                ]}
                with open(file_path, 'w') as f:
                    json.dump(config, f)
                print(f"[INFO] Config saved: {file_path}")
            except Exception as e:
                print(f"[ERROR] {e}")

    def _handle_pairing_click(self, mouse_pos, button):
        x, y = mouse_pos
        grid_x = int(x / self.cell_w)
        grid_y = int(y / self.cell_h)

        if not (0 <= grid_x < self.map_width and 0 <= grid_y < self.map_height):
            return

        cell = self.city_map[grid_y, grid_x]
        pos = (grid_x, grid_y)

        paired_spawns = {p['spawn'] for p in self.pairs}
        paired_finishes = {p['finish'] for p in self.pairs}

        if button == 1:
            if cell == 2 and pos not in paired_spawns:
                self.selected_spawn = pos
                print(f"[INFO] Spawn selected: {pos}")

            elif cell == 4 and self.selected_spawn is not None and pos not in paired_finishes:
                self.pairs.append({'spawn': self.selected_spawn, 'finish': pos})
                print(f"[INFO] Pair {len(self.pairs) - 1}: spawn {self.selected_spawn} → finish {pos}")
                self.selected_spawn = None

        elif button == 3:
            self.pairs = [p for p in self.pairs if p['spawn'] != pos and p['finish'] != pos]
            if self.selected_spawn == pos:
                self.selected_spawn = None
            print(f"[INFO] Pair removed at {pos}")

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.VIDEORESIZE:
                self.win_w, self.win_h = event.w, event.h
                self._update_metrics()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_0:
                    self.current_block = 0
                elif event.key == pygame.K_1:
                    self.current_block = 1
                elif event.key == pygame.K_2:
                    self.current_block = 2
                elif event.key == pygame.K_3:
                    self.current_block = 3
                elif event.key == pygame.K_4:
                    self.current_block = 4
                elif event.key == pygame.K_s:
                    self._save_map()
                elif event.key == pygame.K_l:
                    self._load_map()
                elif event.key == pygame.K_r:
                    self._resize_map()
                elif event.key == pygame.K_c:
                    self.city_map.fill(0)
                    print("[INFO] Map has been cleared")
                elif event.key == pygame.K_p:
                    self.pairing_mode = not self.pairing_mode
                    self.selected_spawn = None
                    print(f"[INFO] Pairing mode: {'ON' if self.pairing_mode else 'OFF'}")
                elif event.key == pygame.K_j:
                    self._save_config()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.pairing_mode:
                    self._handle_pairing_click(event.pos, event.button)
                else:
                    if event.button == 1:
                        self.drawing = True
                    elif event.button == 3:
                        self.erasing = True
                    self._apply_brush(event.pos)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.drawing = False
                elif event.button == 3:
                    self.erasing = False

            elif event.type == pygame.MOUSEMOTION:
                if self.drawing or self.erasing:
                    self._apply_brush(event.pos)

        return True

    def _render_pairs(self):
        paired_spawns = {p['spawn']: i for i, p in enumerate(self.pairs)}
        paired_finishes = {p['finish']: i for i, p in enumerate(self.pairs)}

        for y in range(self.map_height):
            for x in range(self.map_width):
                pos = (x, y)
                cx = round(x * self.cell_w + self.cell_w / 2)
                cy = round(y * self.cell_h + self.cell_h / 2)

                if pos == self.selected_spawn:
                    rx = round(x * self.cell_w)
                    ry = round(y * self.cell_h)
                    pygame.draw.rect(self.window, (255, 255, 255),
                                     pygame.Rect(rx, ry, round(self.cell_w), round(self.cell_h)), 3)

                if pos in paired_spawns:
                    txt = self.pair_font.render(str(paired_spawns[pos]), True, (0, 0, 0))
                    self.window.blit(txt, txt.get_rect(center=(cx, cy)))
                elif pos in paired_finishes:
                    txt = self.pair_font.render(str(paired_finishes[pos]), True, (255, 255, 255))
                    self.window.blit(txt, txt.get_rect(center=(cx, cy)))

    def _render(self):
        if not self.window:
            return

        self.window.fill((30, 30, 30))

        for y in range(self.map_height):
            for x in range(self.map_width):
                color = self.cell_colors.get(self.city_map[y, x], (0, 0, 0))
                x0 = round(x * self.cell_w)
                y0 = round(y * self.cell_h)
                x1 = round((x + 1) * self.cell_w)
                y1 = round((y + 1) * self.cell_h)
                rect = pygame.Rect(x0, y0, x1 - x0, y1 - y0)
                pygame.draw.rect(self.window, color, rect)
                pygame.draw.rect(self.window, (50, 50, 50), rect, 1)

        self._render_pairs()

        ui_y = self.win_h - UI_PANEL_HEIGHT
        pygame.draw.rect(self.window, (20, 20, 20), pygame.Rect(0, ui_y, self.win_w, UI_PANEL_HEIGHT))

        if not self.pairing_mode:
            line1 = f"Brush: {self.block_names[self.current_block]}"
            line2 = "0-4: Select | S: Save | L: Load | R: Resize | C: Clear | P: Pairing mode"
        else:
            line1 = f"PAIRING MODE | Pairs: {len(self.pairs)}"
            line2 = "LMB: select spawn, then finish | RMB: remove pair | J: Save JSON | P: Exit"

        self.window.blit(self.font.render(line1, True, (220, 220, 220)), (10, ui_y + 6))
        self.window.blit(self.font.render(line2, True, (130, 130, 130)), (10, ui_y + 28))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            self.clock.tick(60)
            running = self._handle_events()
            self._render()

        pygame.quit()
        sys.exit()