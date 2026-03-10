"""
config.py — Hiperparâmetros centralizados do projeto.

Altere os valores aqui para ajustar o treinamento sem mexer no código.
"""
import os

# ===========================
# CAMINHOS
# ===========================
ROM_PATH = "Pokemon_FireRed.gb"
SAVE_STATE_PATH = "checkpoint.state"        # Save state do emulador para reset rápido
MODEL_DIR = "models"                        # Pasta onde os modelos treinados são salvos
LOG_DIR = "logs"                            # Pasta dos logs do TensorBoard

# ===========================
# EMULADOR
# ===========================
FRAMES_PER_ACTION = 24      # Quantos frames avançar por ação (24 ≈ 1 tile de movimento)
MAX_STEPS_PER_EPISODE = 2048 # Máximo de ações por episódio antes de resetar

# ===========================
# HIPERPARÂMETROS DO PPO
# ===========================
LEARNING_RATE = 2.5e-4      # Taxa de aprendizado
N_STEPS = 2048              # Steps coletados antes de cada atualização
BATCH_SIZE = 64             # Tamanho do mini-batch
N_EPOCHS = 4                # Épocas por atualização
GAMMA = 0.998               # Fator de desconto (alto = valoriza recompensas futuras)
GAE_LAMBDA = 0.95           # Generalized Advantage Estimation
CLIP_RANGE = 0.2            # Clip do PPO
ENT_COEF = 0.01             # Coeficiente de entropia (incentiva exploração)
TOTAL_TIMESTEPS = 1_000_000 # Total de timesteps de treino (ajuste conforme desejado)

# Rede Neural do agente (MLP = Multi-Layer Perceptron)
POLICY_KWARGS = dict(
    net_arch=dict(
        pi=[128, 64],       # Rede da política (decisão de ações)
        vf=[128, 64],       # Rede do valor (estimativa de recompensa futura)
    )
)

# ===========================
# RECOMPENSAS
# ===========================
REWARD_NEW_TILE = 0.02       # Recompensa por visitar um tile novo
REWARD_NEW_MAP = 2.0         # Recompensa por entrar em um mapa nunca visitado
REWARD_BADGE = 50.0          # Recompensa por ganhar uma badge
REWARD_LEVEL_UP = 5.0        # Recompensa por level up do pokémon
REWARD_HP_LOSS_PENALTY = -0.5  # Penalidade por perda de HP (por % perdido)
REWARD_STUCK_PENALTY = -0.1  # Penalidade por ficar travado (mesma posição por muito tempo)
REWARD_BLACKOUT = -10.0      # Penalidade por blackout (todos os pokémon morrem)

STUCK_THRESHOLD = 20         # Quantos steps parado antes de começar a penalizar

# ===========================
# LOGGING
# ===========================
SAVE_FREQUENCY = 50_000     # Salvar modelo a cada N timesteps
LOG_FREQUENCY = 1           # Frequência de log no TensorBoard (1 = todo episódio)
VERBOSE = 1                 # 0=silencioso, 1=infos, 2=debug

# Cria pastas se não existirem
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
