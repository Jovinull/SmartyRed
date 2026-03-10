"""
memory_map.py — Endereços de Memória RAM do Pokémon Red/Blue (Game Boy)

Referência: https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
Todos os endereços são do espaço WRAM (Work RAM) do Game Boy.
"""

# ===========================
# POSIÇÃO DO JOGADOR
# ===========================
PLAYER_Y = 0xD361            # Posição Y no mapa atual (0-255)
PLAYER_X = 0xD362            # Posição X no mapa atual (0-255)
PLAYER_DIRECTION = 0xC109    # Direção que o jogador está olhando (0=baixo, 4=cima, 8=esq, 0xC=dir)

# ===========================
# MAPA
# ===========================
MAP_ID = 0xD35E              # ID do mapa atual (0=Pallet Town, 1=Viridian City, etc.)
MAP_HEIGHT = 0xD368          # Altura do mapa atual em tiles
MAP_WIDTH = 0xD369           # Largura do mapa atual em tiles

# ===========================
# ESTADO DE BATALHA
# ===========================
BATTLE_TYPE = 0xD057         # 0=Sem batalha, 1=Selvagem, 2=Treinador, -1=Perdeu
IS_IN_BATTLE = 0xD057        # Mesmo endereço, alias semântico

# Pokémon inimigo em batalha
ENEMY_POKEMON_HP_HIGH = 0xCFE6
ENEMY_POKEMON_HP_LOW = 0xCFE7
ENEMY_POKEMON_LEVEL = 0xCFF3

# ===========================
# PARTY DO JOGADOR
# ===========================
PARTY_COUNT = 0xD163         # Quantidade de Pokémon na party (1-6)

# Pokémon 1 da Party (offsets para os demais: +44 bytes cada)
PARTY_POKEMON_SIZE = 44      # Tamanho de cada struct de Pokémon na party

PARTY1_SPECIES = 0xD164      # ID da espécie do Pokémon 1
PARTY1_HP_HIGH = 0xD16C      # HP atual (byte alto)
PARTY1_HP_LOW = 0xD16D       # HP atual (byte baixo)
PARTY1_LEVEL = 0xD18C        # Level do Pokémon 1
PARTY1_MAX_HP_HIGH = 0xD18D  # HP máximo (byte alto)
PARTY1_MAX_HP_LOW = 0xD18E   # HP máximo (byte baixo)
PARTY1_XP_HIGH = 0xD179      # XP (byte alto)
PARTY1_XP_MID = 0xD17A       # XP (byte médio)
PARTY1_XP_LOW = 0xD17B       # XP (byte baixo)

# ===========================
# PROGRESSO DO JOGO
# ===========================
BADGES = 0xD356              # Bitfield de badges (bit 0=Boulder, bit 1=Cascade, etc.)
MONEY_HIGH = 0xD347          # Dinheiro (BCD, byte alto)
MONEY_MID = 0xD348           # Dinheiro (BCD, byte médio)
MONEY_LOW = 0xD349           # Dinheiro (BCD, byte baixo)

# Pokédex
POKEDEX_OWNED_START = 0xD2F7 # 19 bytes, cada bit é 1 Pokémon visto
POKEDEX_OWNED_BYTES = 19     # 19 bytes * 8 bits = 152 Pokémon

# ===========================
# MENU E UI
# ===========================
TEXT_BOX_OPEN = 0xD125       # Se != 0, uma caixa de texto está aberta
MENU_ITEM_ID = 0xCC26        # Item selecionado no menu atual

# ===========================
# NOMES DOS MAPAS (para logs legíveis)
# ===========================
MAP_NAMES = {
    0: "Pallet Town",
    1: "Viridian City",
    2: "Pewter City",
    3: "Cerulean City",
    4: "Lavender Town",
    5: "Vermilion City",
    6: "Celadon City",
    7: "Fuchsia City",
    8: "Cinnabar Island",
    9: "Indigo Plateau",
    10: "Saffron City",
    12: "Route 1",
    13: "Route 2",
    14: "Route 3",
    15: "Route 4",
    33: "Route 22",
    37: "Red's House 1F",
    38: "Red's House 2F",
    39: "Blue's House",
    40: "Oak's Lab",
}

# Nomes dos badges
BADGE_NAMES = [
    "Boulder Badge",
    "Cascade Badge",
    "Thunder Badge",
    "Rainbow Badge",
    "Soul Badge",
    "Marsh Badge",
    "Volcano Badge",
    "Earth Badge",
]
