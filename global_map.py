"""
global_map.py — Converte coordenadas locais (x, y, map_id) em globais.

Usa map_data.json para posicionar cada mapa do jogo num espaço 2D unificado.
Adaptado de: https://github.com/thatguy11325/pokemonred_puffer
"""
import os
import json

MAP_PATH = os.path.join(os.path.dirname(__file__), "map_data.json")
PAD = 20
GLOBAL_MAP_SHAPE = (444 + PAD * 2, 436 + PAD * 2)
MAP_ROW_OFFSET = PAD
MAP_COL_OFFSET = PAD

with open(MAP_PATH) as map_data:
    MAP_DATA = json.load(map_data)["regions"]
MAP_DATA = {int(e["id"]): e for e in MAP_DATA}


def local_to_global(r: int, c: int, map_n: int):
    """Converte coordenadas locais do jogo para coordenadas globais."""
    try:
        map_x, map_y = MAP_DATA[map_n]["coordinates"]
        gy = r + map_y + MAP_ROW_OFFSET
        gx = c + map_x + MAP_COL_OFFSET
        if 0 <= gy < GLOBAL_MAP_SHAPE[0] and 0 <= gx < GLOBAL_MAP_SHAPE[1]:
            return gy, gx
        return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2
    except KeyError:
        return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2
