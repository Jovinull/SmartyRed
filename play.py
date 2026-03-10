"""
play.py — Assistir o agente treinado jogar com janela gráfica.

Modos de uso:
    python play.py                    → Assiste o melhor modelo treinado jogar
    python play.py --model models/pokemon_ppo_final.zip  → Carrega modelo específico
    python play.py --create-save      → Jogue manualmente para criar um save state
"""
import argparse
import os
import time

from stable_baselines3 import PPO
from pyboy.utils import WindowEvent

import config
from environment import PokemonRedEnv, ACTION_NAMES
import memory_map as mem


def find_best_model() -> str | None:
    """Encontra o modelo mais recente/melhor na pasta de modelos."""
    final = os.path.join(config.MODEL_DIR, "pokemon_ppo_final.zip")
    if os.path.exists(final):
        return final

    if not os.path.exists(config.MODEL_DIR):
        return None

    models = [
        f for f in os.listdir(config.MODEL_DIR)
        if f.startswith("pokemon_ppo_") and f.endswith(".zip")
    ]
    if not models:
        return None

    models.sort(key=lambda x: int(x.replace("pokemon_ppo_", "").replace("_steps.zip", "")))
    return os.path.join(config.MODEL_DIR, models[-1])


def create_save_state():
    """
    Permite jogar manualmente para criar o save state inicial.
    O usuário deve jogar até escolher seu Pokémon inicial e salvar.
    """
    print("=" * 60)
    print("   MODO: CRIAR SAVE STATE")
    print("=" * 60)
    print("  Jogue normalmente até ter seu primeiro Pokémon.")
    print("  Quando estiver pronto, feche a janela do jogo.")
    print("  O estado será salvo automaticamente.\n")

    from pyboy import PyBoy
    pyboy = PyBoy(config.ROM_PATH, window="SDL2")
    pyboy.set_emulation_speed(1)

    try:
        while pyboy.tick():
            pass
    except KeyboardInterrupt:
        pass

    # Salva o estado
    with open(config.SAVE_STATE_PATH, "wb") as f:
        pyboy.save_state(f)
    print(f"\n[OK] Save state criado: {config.SAVE_STATE_PATH}")

    # Mostra info do estado salvo
    m = pyboy.memory
    party_count = m[mem.PARTY_COUNT]
    level = m[mem.PARTY1_LEVEL]
    map_id = m[mem.MAP_ID]
    map_name = mem.MAP_NAMES.get(map_id, f"Unknown({map_id})")
    print(f"  Pokémon na party: {party_count}")
    print(f"  Level do primeiro: {level}")
    print(f"  Mapa: {map_name}")

    pyboy.stop()


def watch_agent(model_path: str):
    """Abre a janela do jogo e assiste o agente treinado jogar."""
    print("=" * 60)
    print("   POKÉMON RED — ASSISTINDO O AGENTE JOGAR")
    print("=" * 60)
    print(f"  Modelo: {model_path}\n")

    # Cria ambiente COM janela gráfica
    env = PokemonRedEnv(render_mode="human")
    env.pyboy.set_emulation_speed(1)  # Velocidade normal para visualização

    # Carrega o modelo treinado
    model = PPO.load(model_path, env=env)

    obs, _ = env.reset()
    total_reward = 0
    step_count = 0

    print("[PLAY] Agente jogando! Feche a janela ou Ctrl+C para parar.\n")

    try:
        while True:
            # O agente escolhe uma ação
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))

            total_reward += reward
            step_count += 1

            # Log periódico
            if step_count % 50 == 0:
                print(
                    f"  Step {step_count:>5} | "
                    f"Reward: {total_reward:>8.2f} | "
                    f"Tiles: {info.get('tiles_explored', 0):>4} | "
                    f"Mapas: {info.get('maps_explored', 0):>2} | "
                    f"Badges: {info.get('badges', 0)} | "
                    f"Mapa: {info.get('map_name', '?')}"
                )

            if terminated or truncated:
                print(f"\n[EP FIM] Reward Total: {total_reward:.2f} | Steps: {step_count}")
                obs, _ = env.reset()
                total_reward = 0
                step_count = 0

    except KeyboardInterrupt:
        print("\n[PLAY] Interrompido pelo usuário.")
    except SystemExit:
        pass
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Assistir o Agente Pokémon jogar")
    parser.add_argument("--model", type=str, default=None, help="Caminho do modelo .zip")
    parser.add_argument("--create-save", action="store_true", help="Jogar manualmente para criar save state")
    args = parser.parse_args()

    if args.create_save:
        create_save_state()
        return

    # Encontra o modelo
    model_path = args.model or find_best_model()
    if model_path is None or not os.path.exists(model_path):
        print("[ERRO] Nenhum modelo treinado encontrado!")
        print("       Execute 'python train.py' primeiro para treinar o agente.")
        print("       Ou use --model para especificar o caminho de um modelo .zip")
        return

    watch_agent(model_path)


if __name__ == "__main__":
    main()
