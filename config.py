"""
config.py — Hiperparâmetros centralizados.

Valores adaptados do PokemonRedExperiments V2 para execução single-CPU.
O projeto original usa 64 CPUs. Para i5 13ª gen, usamos 1 CPU com episódios mais longos.
"""
import os

# ===========================
# CAMINHOS
# ===========================
ROM_PATH = "Pokemon_FireRed.gb"
INIT_STATE_PATH = "init.state"            # Save state inicial (início do jogo)
MODEL_DIR = "models"
LOG_DIR = "logs"
SESSION_DIR = "session"

# ===========================
# EMULADOR
# ===========================
ACTION_FREQ = 24            # Frames por ação (padrão do projeto referência)
# No projeto original: ep_length = 2048 * 80 = 163840 com 64 CPUs
# Para 1 CPU: usamos episódios mais curtos para ter feedback mais rápido
MAX_STEPS_PER_EPISODE = 2048 * 40  # ~81920 steps por episódio

# ===========================
# HIPERPARÂMETROS DO PPO
# Adaptados do baseline_fast_v2.py
# ===========================
# O ref usa: n_steps = ep_length // 64, batch=512, epochs=1, gamma=0.997, ent=0.01
NUM_ENVS = 1                # Número de ambientes paralelos (1 para i5)
N_STEPS = 2048              # Steps coletados antes de cada atualização
BATCH_SIZE = 512            # Tamanho do mini-batch
N_EPOCHS = 1                # 1 época por atualização (como o referência)
GAMMA = 0.997               # Fator de desconto (referência usa 0.997)
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01             # Coeficiente de entropia
LEARNING_RATE = 2.5e-4
TOTAL_TIMESTEPS = 10_000_000  # Treina por 10M steps (ajuste conforme quiser)

# Rede Neural — usa MultiInputPolicy (Dict observation)
# O referência usa a rede padrão do SB3 para MultiInput

# ===========================
# RECOMPENSAS (Adaptadas do referência)
# ===========================
REWARD_SCALE = 0.5          # Escala global das recompensas
EXPLORE_WEIGHT = 0.25       # Peso da exploração

# Pesos individuais (multiplicados por reward_scale internamente)
EVENT_REWARD_MULT = 4.0     # Multiplicador para event flags
HEAL_REWARD_MULT = 10.0     # Multiplicador para cura
BADGE_REWARD_MULT = 10.0    # Multiplicador por badge
EXPLORE_REWARD_MULT = 0.1   # Multiplicador por tile explorado
STUCK_PENALTY_MULT = -0.05  # Penalidade por ficar preso (>600 visitas ao mesmo tile)
STUCK_VISIT_THRESHOLD = 600 # Quantas visitas ao mesmo tile = considerado "preso"

# ===========================
# OBSERVAÇÃO
# ===========================
FRAME_STACKS = 3            # Quantos frames empilhar (memória visual)
SCREEN_SIZE = (72, 80)      # Tamanho da tela reduzida
COORDS_PAD = 12             # Raio do mapa de exploração local
ENC_FREQS = 8               # Frequências para Fourier encoding dos levels

# ===========================
# LOGGING
# ===========================
SAVE_FREQUENCY = 50_000
SAVE_VIDEO = False
FAST_VIDEO = True
PRINT_REWARDS = True
VERBOSE = 1

# Cria pastas
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)
