"""
memory_map.py — Endereços de Memória RAM do Pokémon Red/Blue (Game Boy)

Referência: https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
Adaptado do PokemonRedExperiments por Peter Whidden.
"""

# ===========================
# POSIÇÃO DO JOGADOR
# ===========================
PLAYER_Y = 0xD361
PLAYER_X = 0xD362
PLAYER_DIRECTION = 0xC109

# ===========================
# MAPA
# ===========================
MAP_ID = 0xD35E
MAP_HEIGHT = 0xD368
MAP_WIDTH = 0xD369

# ===========================
# ESTADO DE BATALHA
# ===========================
BATTLE_TYPE = 0xD057  # 0=nenhuma, 1=selvagem, 2=treinador

# ===========================
# PARTY DO JOGADOR (6 slots, 44 bytes cada)
# ===========================
PARTY_COUNT = 0xD163
PARTY_POKEMON_SIZE = 44

# Endereços base de cada slot da party (espécie)
PARTY_SPECIES = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]

# HP de cada pokémon na party (2 bytes cada: high, then low)
PARTY_HP = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]

# HP máximo de cada pokémon
PARTY_MAX_HP = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]

# Level de cada pokémon
PARTY_LEVELS = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]

# XP (3 bytes cada)
PARTY_XP = [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]

# ===========================
# OPONENTE EM BATALHA
# ===========================
OPPONENT_LEVELS = [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]

# ===========================
# PROGRESSO DO JOGO
# ===========================
BADGES = 0xD356

# Event flags — cada bit representa um evento do jogo
# (falou com NPC tal, pegou item tal, derrotou treinador tal)
EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END = 0xD87E  # Expandido para incluir SS Anne
MUSEUM_TICKET = (0xD754, 0)  # Flag do ticket do museu (ignorar na contagem)

# ===========================
# MAPAS ESSENCIAIS PARA PROGRESSÃO
# Índice = prioridade na progressão do jogo
# ===========================
ESSENTIAL_MAP_LOCATIONS = {
    40: 0,   # Oak's Lab
    0: 1,    # Pallet Town
    12: 2,   # Route 1
    1: 3,    # Viridian City
    13: 4,   # Route 2
    51: 5,   # Viridian Forest
    2: 6,    # Pewter City
    54: 7,   # Pewter Gym? / Mt Moon?
    14: 8,   # Route 3
    59: 9,   # Mt Moon 1F
    60: 10,  # Mt Moon B1F
    61: 11,  # Mt Moon B2F
    15: 12,  # Route 4
    3: 13,   # Cerulean City
    65: 14,  # Cerulean Gym
}
