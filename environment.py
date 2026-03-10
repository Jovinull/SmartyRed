"""
environment.py — Gymnasium Environment que envolve o PyBoy.

Transforma o emulador do Pokémon Red num "ginásio de treino" para RL.
O agente recebe uma observação vetorial (dados da RAM) e escolhe 1 de 7 botões.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import os

import memory_map as mem
import config
from rewards import RewardCalculator


# Mapeamento: índice da ação → (press_event, release_event)
ACTIONS = [
    (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
    (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
    (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
    (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
    (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
    (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
    (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START),
]

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START"]


class PokemonRedEnv(gym.Env):
    """
    Gymnasium Environment para Pokémon Red via PyBoy.

    Observation Space:
        Box(10,) com valores normalizados entre 0 e 1:
        [player_x, player_y, map_id, direction, battle_state,
         hp_ratio, level, badges_ratio, party_count, enemy_hp_ratio]

    Action Space:
        Discrete(7): UP, DOWN, LEFT, RIGHT, A, B, START
    """

    metadata = {"render_modes": ["human", "null"], "render_fps": 60}

    def __init__(self, render_mode="null"):
        super().__init__()

        self.render_mode = render_mode
        self.rom_path = config.ROM_PATH

        # Espaço de ações: 7 botões
        self.action_space = spaces.Discrete(len(ACTIONS))

        # Espaço de observação: vetor com 10 valores normalizados [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )

        # Inicializa o emulador
        window = "SDL2" if render_mode == "human" else "null"
        self.pyboy = PyBoy(self.rom_path, window=window)

        if render_mode != "human":
            self.pyboy.set_emulation_speed(0)  # Velocidade máxima (sem limite de FPS)

        # Sistema de recompensas
        self.reward_calc = RewardCalculator()

        # Controle de episódio
        self.current_step = 0
        self.has_save_state = os.path.exists(config.SAVE_STATE_PATH)

        # Cria o save state inicial se não existir
        if not self.has_save_state:
            print("[ENV] Save state não encontrado. Criando a partir do estado atual...")
            print("[ENV] Dica: Jogue até escolher seu Pokémon inicial e salve com play.py")
            # Avança alguns frames para sair da tela preta
            for _ in range(60):
                self.pyboy.tick()
            self._save_state()
        else:
            self._load_state()

    def _save_state(self):
        """Salva o estado atual do emulador para disco."""
        with open(config.SAVE_STATE_PATH, "wb") as f:
            self.pyboy.save_state(f)
        self.has_save_state = True
        print(f"[ENV] Save state salvo em: {config.SAVE_STATE_PATH}")

    def _load_state(self):
        """Carrega o save state do disco."""
        with open(config.SAVE_STATE_PATH, "rb") as f:
            self.pyboy.load_state(f)

    def _read_ram(self) -> dict:
        """Lê os valores relevantes da memória RAM."""
        m = self.pyboy.memory

        # Posição
        player_x = m[mem.PLAYER_X]
        player_y = m[mem.PLAYER_Y]
        map_id = m[mem.MAP_ID]
        direction = m[mem.PLAYER_DIRECTION]

        # Batalha
        battle_state = m[mem.BATTLE_TYPE]

        # Party Pokémon 1
        hp_current = (m[mem.PARTY1_HP_HIGH] << 8) | m[mem.PARTY1_HP_LOW]
        hp_max = (m[mem.PARTY1_MAX_HP_HIGH] << 8) | m[mem.PARTY1_MAX_HP_LOW]
        level = m[mem.PARTY1_LEVEL]
        party_count = m[mem.PARTY_COUNT]

        # Level total da party (soma de todos os pokémon)
        total_level = 0
        for i in range(min(party_count, 6)):
            addr = mem.PARTY1_LEVEL + (mem.PARTY_POKEMON_SIZE * i)
            total_level += m[addr]

        # Badges: conta os bits ativos
        badges_byte = m[mem.BADGES]
        badges_count = bin(badges_byte).count("1")

        # Inimigo (se em batalha)
        enemy_hp = 0
        if battle_state > 0:
            enemy_hp = (m[mem.ENEMY_POKEMON_HP_HIGH] << 8) | m[mem.ENEMY_POKEMON_HP_LOW]

        return {
            "player_x": player_x,
            "player_y": player_y,
            "map_id": map_id,
            "direction": direction,
            "battle_state": battle_state,
            "hp_current": hp_current,
            "hp_max": hp_max,
            "level": level,
            "total_level": total_level,
            "party_count": party_count,
            "badges_count": badges_count,
            "enemy_hp": enemy_hp,
        }

    def _get_observation(self, state: dict) -> np.ndarray:
        """Converte os dados da RAM em um vetor normalizado [0, 1]."""
        hp_ratio = state["hp_current"] / max(state["hp_max"], 1)
        enemy_hp_ratio = min(state["enemy_hp"] / 200.0, 1.0)  # Normaliza aproximadamente

        obs = np.array([
            state["player_x"] / 255.0,
            state["player_y"] / 255.0,
            state["map_id"] / 255.0,
            state["direction"] / 12.0,       # Direção: 0, 4, 8, 12
            state["battle_state"] / 2.0,
            hp_ratio,
            state["level"] / 100.0,
            state["badges_count"] / 8.0,
            state["party_count"] / 6.0,
            enemy_hp_ratio,
        ], dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)

    def _press_button(self, action_idx: int):
        """Pressiona e solta um botão avançando frames intermediários."""
        press_event, release_event = ACTIONS[action_idx]

        # Pressiona
        self.pyboy.send_input(press_event)

        # Mantém pressionado por alguns frames
        for _ in range(8):
            self.pyboy.tick()

        # Solta
        self.pyboy.send_input(release_event)

        # Avança frames restantes para completar a ação
        for _ in range(config.FRAMES_PER_ACTION - 8):
            self.pyboy.tick()

    def reset(self, seed=None, options=None):
        """Reseta o ambiente para o início de um novo episódio."""
        super().reset(seed=seed)

        # Recarrega o save state
        if self.has_save_state:
            self._load_state()

        # Reseta o sistema de recompensas
        self.reward_calc.reset()
        self.current_step = 0

        # Avança poucos frames para estabilizar após o load
        for _ in range(10):
            self.pyboy.tick()

        state = self._read_ram()
        obs = self._get_observation(state)

        # Inicializa o rastreamento com o estado inicial
        self.reward_calc.last_position = (
            state["player_x"], state["player_y"], state["map_id"]
        )
        self.reward_calc.total_level = state["total_level"]
        self.reward_calc.badges_count = state["badges_count"]
        if state["hp_max"] > 0:
            self.reward_calc.last_hp_fraction = state["hp_current"] / state["hp_max"]

        return obs, {}

    def step(self, action: int):
        """Executa uma ação e retorna (obs, reward, terminated, truncated, info)."""
        # 1. Executa a ação no emulador
        self._press_button(action)
        self.current_step += 1

        # 2. Lê o estado da RAM
        state = self._read_ram()
        obs = self._get_observation(state)

        # 3. Calcula a recompensa
        reward, is_blackout = self.reward_calc.calculate(state)

        # 4. Verifica fim do episódio
        terminated = is_blackout
        truncated = self.current_step >= config.MAX_STEPS_PER_EPISODE

        # 5. Info para logging
        info = self.reward_calc.get_stats()
        info["action"] = ACTION_NAMES[action]
        info["map_name"] = mem.MAP_NAMES.get(state["map_id"], f"Unknown({state['map_id']})")

        return obs, reward, terminated, truncated, info

    def render(self):
        """Rendering é tratado automaticamente pelo PyBoy quando window='SDL2'."""
        pass

    def close(self):
        """Fecha o emulador."""
        try:
            self.pyboy.stop()
        except Exception:
            pass
