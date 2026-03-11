"""
environment.py — Gymnasium Environment para Pokémon Red via PyBoy.

Adaptado do PokemonRedExperiments V2 (Peter Whidden).
Usa observação multimodal (Dict), event flags como recompensa de progressão,
sistema de cura/morte inteligente, e mapa global de exploração.
"""
import json
import os
from pathlib import Path

import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import gymnasium as gym
from gymnasium import spaces

import memory_map as mem
import config
from global_map import local_to_global, GLOBAL_MAP_SHAPE


# Ações disponíveis
VALID_ACTIONS = [
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
    WindowEvent.PRESS_BUTTON_START,
]

RELEASE_ACTIONS = [
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.RELEASE_BUTTON_START,
]

ACTION_NAMES = ["DOWN", "LEFT", "RIGHT", "UP", "A", "B", "START"]


class PokemonRedEnv(gym.Env):
    """
    Gymnasium Environment para treinar um agente via RL a jogar Pokémon Red.

    Observação (Dict):
        - screens: 3 frames empilhados em escala de cinza (72, 80, 3)
        - health: fração de HP total da party [0, 1]
        - level: encoding Fourier da soma dos levels
        - badges: vetor binário de 8 badges
        - events: vetor binário de todas as event flags do jogo
        - map: mapa local de exploração (48x48)
        - recent_actions: últimas 3 ações tomadas

    Recompensa (baseada no PokemonRedExperiments V2):
        - Event flags ativadas (progresso na história)
        - Cura no Pokémon Center
        - Badges conquistadas
        - Tiles explorados
        - Penalidade por ficar travado (>600 visitas)
    """

    metadata = {"render_modes": ["human", "null"]}

    def __init__(self, render_mode="null"):
        super().__init__()

        self.render_mode = render_mode
        self.headless = render_mode != "human"

        # Carrega nomes dos eventos
        events_path = os.path.join(os.path.dirname(__file__), "events.json")
        with open(events_path) as f:
            self.event_names = json.load(f)

        # Calcula dimensões
        self.frame_stacks = config.FRAME_STACKS
        self.output_shape = (config.SCREEN_SIZE[0], config.SCREEN_SIZE[1], self.frame_stacks)
        self.coords_pad = config.COORDS_PAD
        self.enc_freqs = config.ENC_FREQS
        self.n_event_flags = (mem.EVENT_FLAGS_END - mem.EVENT_FLAGS_START) * 8

        # Espaço de ações
        self.action_space = spaces.Discrete(len(VALID_ACTIONS))

        # Espaço de observação multimodal (Dict)
        self.observation_space = spaces.Dict({
            "screens": spaces.Box(
                low=0, high=255, shape=self.output_shape, dtype=np.uint8
            ),
            "health": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "level": spaces.Box(
                low=-1, high=1, shape=(self.enc_freqs,), dtype=np.float32
            ),
            "badges": spaces.MultiBinary(8),
            "events": spaces.MultiBinary(self.n_event_flags),
            "map": spaces.Box(
                low=0, high=255,
                shape=(self.coords_pad * 4, self.coords_pad * 4, 1),
                dtype=np.uint8,
            ),
            "recent_actions": spaces.MultiDiscrete(
                [len(VALID_ACTIONS)] * self.frame_stacks
            ),
        })

        # Inicia o emulador
        window = "null" if self.headless else "SDL2"
        self.pyboy = PyBoy(
            config.ROM_PATH,
            window=window,
            sound_emulated=not self.headless,  # Desliga som no treino (evita buffer overrun)
        )
        if self.headless:
            self.pyboy.set_emulation_speed(0)  # Velocidade máxima sem som
        else:
            self.pyboy.set_emulation_speed(6)

        # Estado interno (será populado no reset)
        self.step_count = 0
        self.reset_count = 0

    def reset(self, seed=None, options=None):
        """Reseta o ambiente carregando o save state inicial."""
        super().reset(seed=seed)

        # Carrega o save state
        with open(config.INIT_STATE_PATH, "rb") as f:
            self.pyboy.load_state(f)

        # Reseta rastreamento
        self.seen_coords = {}
        self.explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.uint8)
        self.recent_screens = np.zeros(self.output_shape, dtype=np.uint8)
        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1.0
        self.total_healing_rew = 0.0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0
        self.max_map_progress = 0
        self.current_event_flags_set = {}

        # Calcula base de event flags (para só contar as novas)
        self.base_event_flags = sum(
            self._bit_count(self._read_m(i))
            for i in range(mem.EVENT_FLAGS_START, mem.EVENT_FLAGS_END)
        )

        # Calcula recompensa inicial
        self.progress_reward = self._get_game_state_reward()
        self.total_reward = sum(self.progress_reward.values())

        self.reset_count += 1
        return self._get_obs(), {}

    # ==========================
    # STEP
    # ==========================

    def step(self, action):
        """Executa uma ação e retorna (obs, reward, terminated, truncated, info)."""
        # 1. Executa a ação no emulador
        self._run_action(action)

        # 2. Atualiza tracking
        self._update_recent_actions(action)
        self._update_seen_coords()
        self._update_explore_map()
        self._update_heal_reward()

        self.party_size = self._read_m(mem.PARTY_COUNT)

        # 3. Calcula recompensa (delta do total cumulativo)
        new_reward = self._update_reward()

        # 4. Atualiza últimos estados
        self.last_health = self._read_hp_fraction()
        self._update_map_progress()

        # 5. Verifica fim do episódio
        self.step_count += 1
        truncated = self.step_count >= config.MAX_STEPS_PER_EPISODE
        terminated = False  # Nunca termina por morte (como no referência)

        # 6. Info para logging
        info = {
            "step": self.step_count,
            "tiles_explored": len(self.seen_coords),
            "maps_explored": len(set(
                k.split(" m:")[1] for k in self.seen_coords.keys()
            )) if self.seen_coords else 0,
            "badges": self._get_badges(),
            "total_level": self._get_levels_sum(),
            "max_map_progress": self.max_map_progress,
            "deaths": self.died_count,
            "hp": self.last_health,
            "event_reward": self.progress_reward.get("event", 0),
            "healing_reward": self.total_healing_rew,
            "total_reward": self.total_reward,
        }

        obs = self._get_obs()
        return obs, new_reward, terminated, truncated, info

    # ==========================
    # OBSERVAÇÃO
    # ==========================

    def _get_obs(self):
        """Constrói a observação multimodal."""
        screen = self._render_screen()
        self._update_recent_screens(screen)

        # Soma de levels normalizada com encoding Fourier
        level_sum = 0.02 * sum(
            self._read_m(a) for a in mem.PARTY_LEVELS
        )

        # Badges como vetor binário
        badges_byte = self._read_m(mem.BADGES)
        badges = np.array(
            [int(bit) for bit in f"{badges_byte:08b}"], dtype=np.int8
        )

        return {
            "screens": self.recent_screens,
            "health": np.array([self._read_hp_fraction()], dtype=np.float32),
            "level": self._fourier_encode(level_sum),
            "badges": badges,
            "events": np.array(self._read_event_bits(), dtype=np.int8),
            "map": self._get_explore_map_obs(),
            "recent_actions": self.recent_actions,
        }

    def _render_screen(self):
        """Captura a tela do jogo e reduz a resolução."""
        # PyBoy retorna (144, 160, 3). Pegamos apenas 1 canal e reduzimos 2x.
        game_pixels = self.pyboy.screen.ndarray[:, :, 0:1]  # (144, 160, 1)
        # Downscale 2x de forma simples (pega pixels alternados)
        reduced = game_pixels[::2, ::2, :]  # (72, 80, 1)
        return reduced.astype(np.uint8)

    def _update_recent_screens(self, screen):
        """Empilha os últimos N frames."""
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = screen[:, :, 0]

    def _update_recent_actions(self, action):
        """Mantém histórico das últimas N ações."""
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action

    # ==========================
    # EMULADOR
    # ==========================

    def _run_action(self, action):
        """Pressiona um botão e avança frames."""
        self.pyboy.send_input(VALID_ACTIONS[action])
        render = not self.headless
        # Mantém pressionado por 8 frames
        for _ in range(8):
            self.pyboy.tick(1, render)
        # Solta
        self.pyboy.send_input(RELEASE_ACTIONS[action])
        # Avança o restante dos frames
        for _ in range(config.ACTION_FREQ - 8 - 1):
            self.pyboy.tick(1, render)
        self.pyboy.tick(1, True)

    # ==========================
    # LEITURA DA MEMÓRIA
    # ==========================

    def _read_m(self, addr):
        """Lê um byte da memória."""
        return self.pyboy.memory[addr]

    def _read_bit(self, addr, bit):
        """Lê um bit específico de um endereço."""
        return bin(256 + self._read_m(addr))[-bit - 1] == "1"

    def _read_hp(self, start):
        """Lê um valor de HP (2 bytes)."""
        return 256 * self._read_m(start) + self._read_m(start + 1)

    def _read_hp_fraction(self):
        """Lê a fração HP/MaxHP de TODA a party (todos os 6 slots)."""
        hp_sum = sum(self._read_hp(a) for a in mem.PARTY_HP)
        max_hp_sum = sum(self._read_hp(a) for a in mem.PARTY_MAX_HP)
        return hp_sum / max(max_hp_sum, 1)

    def _read_event_bits(self):
        """Lê TODAS as event flags do jogo como vetor binário."""
        return [
            int(bit)
            for addr in range(mem.EVENT_FLAGS_START, mem.EVENT_FLAGS_END)
            for bit in f"{self._read_m(addr):08b}"
        ]

    def _get_game_coords(self):
        """Retorna (x, y, map_id)."""
        return (
            self._read_m(mem.PLAYER_X),
            self._read_m(mem.PLAYER_Y),
            self._read_m(mem.MAP_ID),
        )

    def _get_badges(self):
        """Conta o número de badges."""
        return self._bit_count(self._read_m(mem.BADGES))

    def _read_triple(self, addr):
        """Lê 3 bytes consecutivos (usado para XP)."""
        return 256 * 256 * self._read_m(addr) + 256 * self._read_m(addr + 1) + self._read_m(addr + 2)

    def _get_party_xp(self):
        """Soma o XP atual de toda a party."""
        return sum(self._read_triple(a) for a in mem.PARTY_XP)

    def _bit_count(self, bits):
        """Conta bits ativos."""
        return bin(bits).count("1")

    def _fourier_encode(self, val):
        """Encoding Fourier — converte um escalar em vetor de senos."""
        return np.sin(val * 2 ** np.arange(self.enc_freqs)).astype(np.float32)

    # ==========================
    # EXPLORAÇÃO
    # ==========================

    def _update_seen_coords(self):
        """Rastreia todas as coordenadas visitadas (apenas fora de batalha)."""
        if self._read_m(mem.BATTLE_TYPE) == 0:
            x, y, map_n = self._get_game_coords()
            coord_string = f"x:{x} y:{y} m:{map_n}"
            self.seen_coords[coord_string] = self.seen_coords.get(coord_string, 0) + 1

    def _get_stuck_penalty(self):
        """Retorna 1 se o tile atual foi visitado demais (>600 vezes), 0 caso contrário."""
        x, y, map_n = self._get_game_coords()
        coord_string = f"x:{x} y:{y} m:{map_n}"
        count = self.seen_coords.get(coord_string, 0)
        return 0 if count < config.STUCK_VISIT_THRESHOLD else 1

    def _update_explore_map(self):
        """Marca a posição atual no mapa global de exploração."""
        x, y, map_n = self._get_game_coords()
        gy, gx = local_to_global(y, x, map_n)
        if 0 <= gy < self.explore_map.shape[0] and 0 <= gx < self.explore_map.shape[1]:
            self.explore_map[gy, gx] = 255

    def _get_explore_map_obs(self):
        """Retorna um recorte do mapa de exploração centrado no jogador."""
        x, y, map_n = self._get_game_coords()
        gy, gx = local_to_global(y, x, map_n)
        pad = self.coords_pad

        if 0 <= gy < self.explore_map.shape[0] and 0 <= gx < self.explore_map.shape[1]:
            # Recorte com padding
            y_start = max(gy - pad, 0)
            y_end = min(gy + pad, self.explore_map.shape[0])
            x_start = max(gx - pad, 0)
            x_end = min(gx + pad, self.explore_map.shape[1])
            crop = self.explore_map[y_start:y_end, x_start:x_end]
        else:
            crop = np.zeros((pad * 2, pad * 2), dtype=np.uint8)

        # Resize para (pad*4, pad*4) via repeat
        out = np.repeat(np.repeat(crop, 2, axis=0), 2, axis=1)
        # Garante tamanho exato
        target_h, target_w = pad * 4, pad * 4
        result = np.zeros((target_h, target_w), dtype=np.uint8)
        h = min(out.shape[0], target_h)
        w = min(out.shape[1], target_w)
        result[:h, :w] = out[:h, :w]
        return result[:, :, None]  # (48, 48, 1)

    def _update_map_progress(self):
        """Atualiza o máximo de progresso no mapa (baseado em mapas essenciais)."""
        map_idx = self._read_m(mem.MAP_ID)
        progress = mem.ESSENTIAL_MAP_LOCATIONS.get(map_idx, -1)
        self.max_map_progress = max(self.max_map_progress, progress)

    # ==========================
    # RECOMPENSAS
    # ==========================

    def _update_reward(self):
        """Calcula a reward como delta do total cumulativo (técnica do referência)."""
        self.progress_reward = self._get_game_state_reward()
        new_total = sum(self.progress_reward.values())
        step_reward = new_total - self.total_reward
        self.total_reward = new_total
        return step_reward

    def _get_game_state_reward(self):
        """
        Sistema de recompensas combinando baselines + V2.
        Baselines: level, op_lvl, dead e party_xp (combate).
        V2: event (* 4), stuck penalty, global map (progressão).
        """
        return {
            "event": config.REWARD_SCALE * self._update_max_event_rew() * 4,
            "party_xp": config.REWARD_SCALE * self._get_party_xp() * 0.001,
            "level": config.REWARD_SCALE * self._get_levels_reward(),
            "heal": config.REWARD_SCALE * self.total_healing_rew,
            "op_lvl": config.REWARD_SCALE * self._update_max_op_level(),
            "dead": config.REWARD_SCALE * -0.1 * self.died_count,
            "badge": config.REWARD_SCALE * self._get_badges() * 10,
            "explore": config.REWARD_SCALE * config.EXPLORE_WEIGHT * len(self.seen_coords) * 0.1,
            "stuck": config.REWARD_SCALE * self._get_stuck_penalty() * -0.05,
        }

    def _get_all_events_reward(self):
        """Conta todas as event flags ativadas desde o início (excluindo museum ticket)."""
        total = sum(
            self._bit_count(self._read_m(i))
            for i in range(mem.EVENT_FLAGS_START, mem.EVENT_FLAGS_END)
        )
        museum = int(self._read_bit(mem.MUSEUM_TICKET[0], mem.MUSEUM_TICKET[1]))
        return max(total - self.base_event_flags - museum, 0)

    def _update_max_event_rew(self):
        """Mantém o máximo de event flags ativadas."""
        cur = self._get_all_events_reward()
        self.max_event_rew = max(cur, self.max_event_rew)
        return self.max_event_rew

    def _update_heal_reward(self):
        """
        Detecta cura vs morte (igual ao baselines).
        - Se HP subiu e party_size não mudou → curou (reward = heal_amount * 4)
        - Se HP era 0 e subiu → morreu e reviveu (conta como morte)
        """
        cur_health = self._read_hp_fraction()
        if cur_health > self.last_health and self._read_m(mem.PARTY_COUNT) == self.party_size:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                self.total_healing_rew += heal_amount * 4  # Multiplicador do baselines
            else:
                self.died_count += 1

    def _get_levels_reward(self):
        """
        Recompensa por level up (do baselines).
        Escala linearmente até threshold 22, depois cresce 4x mais devagar.
        Isso incentiva combate no início e evita over-grinding depois.
        """
        explore_thresh = 22
        scale_factor = 4
        level_sum = self._get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum - explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew

    def _update_max_op_level(self):
        """
        Recompensa por enfrentar oponentes mais fortes (do baselines).
        Incentiva a IA a buscar batalhas difíceis.
        """
        opponent_level = max(
            self._read_m(a) for a in mem.OPPONENT_LEVELS
        ) - 5  # Subtrai level base
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level * 0.2

    def _get_levels_sum(self):
        """Soma de levels de todos os pokémon (com ajuste para o level inicial)."""
        min_level = 2
        starter_additional = 4
        levels = [max(self._read_m(a) - min_level, 0) for a in mem.PARTY_LEVELS]
        return max(sum(levels) - starter_additional, 0)

    # ==========================
    # RENDER / CLOSE
    # ==========================

    def render(self):
        """Renderização é tratada pelo SDL2 quando headless=False."""
        return self._render_screen()

    def close(self):
        """Fecha o emulador."""
        try:
            self.pyboy.stop()
        except Exception:
            pass
