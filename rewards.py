"""
rewards.py — Sistema de Recompensas para o Agente de RL.

Rastreia o progresso do jogador e calcula recompensas baseadas em:
- Exploração de tiles/mapas novos
- Badges obtidas
- Level up de Pokémon
- Perda de HP (penalidade)
- Ficar travado (penalidade)
- Blackout (penalidade severa)
"""
import config


class RewardCalculator:
    """Calcula a recompensa a cada step com base no estado anterior vs. atual."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reseta todo o rastreamento de progresso. Chamado a cada episódio."""
        self.visited_tiles = set()     # {(x, y, map_id), ...}
        self.visited_maps = set()      # {map_id, ...}
        self.badges_count = 0          # Quantidade de badges obtidas
        self.total_level = 0           # Soma dos levels de todos os pokémon
        self.last_hp_fraction = 1.0    # HP/HP_Max do pokémon 1 (0.0 a 1.0)
        self.steps_same_position = 0   # Contador de steps parado
        self.last_position = None      # (x, y, map_id) da última ação

    def calculate(self, state: dict) -> float:
        """
        Calcula a recompensa baseada no estado atual do jogo.
        
        Args:
            state: Dicionário com as informações da RAM:
                - player_x, player_y, map_id
                - battle_state
                - hp_current, hp_max
                - level
                - badges_count
                - party_count
                
        Returns:
            float: Recompensa total do step.
        """
        reward = 0.0
        current_pos = (state["player_x"], state["player_y"], state["map_id"])

        # --- 1. Recompensa por exploração ---
        if current_pos not in self.visited_tiles:
            self.visited_tiles.add(current_pos)
            reward += config.REWARD_NEW_TILE

        # --- 2. Recompensa por mapa novo ---
        if state["map_id"] not in self.visited_maps:
            self.visited_maps.add(state["map_id"])
            reward += config.REWARD_NEW_MAP

        # --- 3. Recompensa por badge ---
        if state["badges_count"] > self.badges_count:
            badges_gained = state["badges_count"] - self.badges_count
            reward += config.REWARD_BADGE * badges_gained
            self.badges_count = state["badges_count"]

        # --- 4. Recompensa por level up ---
        if state["total_level"] > self.total_level:
            levels_gained = state["total_level"] - self.total_level
            reward += config.REWARD_LEVEL_UP * levels_gained
            self.total_level = state["total_level"]

        # --- 5. Penalidade por perda de HP ---
        if state["hp_max"] > 0:
            hp_fraction = state["hp_current"] / state["hp_max"]
            if hp_fraction < self.last_hp_fraction:
                hp_lost = self.last_hp_fraction - hp_fraction
                reward += config.REWARD_HP_LOSS_PENALTY * hp_lost
            self.last_hp_fraction = hp_fraction

        # --- 6. Penalidade por ficar travado ---
        if current_pos == self.last_position:
            self.steps_same_position += 1
            if self.steps_same_position > config.STUCK_THRESHOLD:
                reward += config.REWARD_STUCK_PENALTY
        else:
            self.steps_same_position = 0
        self.last_position = current_pos

        # --- 7. Detecção de Blackout (HP do primeiro pokémon zerou) ---
        is_blackout = (
            state["hp_current"] == 0
            and state["party_count"] > 0
            and state["battle_state"] == 0
        )
        if is_blackout:
            reward += config.REWARD_BLACKOUT

        return reward, is_blackout

    def get_stats(self) -> dict:
        """Retorna estatísticas do episódio atual para logging."""
        return {
            "tiles_explored": len(self.visited_tiles),
            "maps_explored": len(self.visited_maps),
            "badges": self.badges_count,
            "total_level": self.total_level,
        }
