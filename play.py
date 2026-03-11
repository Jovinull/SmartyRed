"""
play.py — Assistir o agente jogar ou criar save state.

Uso:
    python play.py                        → Assiste o melhor modelo jogar
    python play.py --model models/x.zip   → Carrega modelo específico
    python play.py --create-save          → Joga manualmente para criar save state
"""
import argparse
import os
import glob
import time

from stable_baselines3 import PPO

import config
from environment import PokemonRedEnv


def find_best_model() -> str | None:
    """Encontra o modelo mais recente."""
    final = os.path.join(config.MODEL_DIR, "pokemon_ppo_final.zip")
    if os.path.exists(final):
        return final

    if not os.path.exists(config.MODEL_DIR):
        return None

    zips = glob.glob(os.path.join(config.MODEL_DIR, "pokemon_ppo_*.zip"))
    if not zips:
        return None
    zips.sort(key=os.path.getmtime)
    return zips[-1]


def create_save_state():
    """Jogue manualmente para criar o save state inicial."""
    from pyboy import PyBoy
    import memory_map as mem

    print("=" * 60)
    print("   MODO: CRIAR SAVE STATE")
    print("=" * 60)
    print("  Jogue até ter seu primeiro Pokémon (escolha o starter).")
    print("  Quando estiver pronto, FECHE A JANELA do jogo.")
    print("  O estado será salvo automaticamente.\n")

    pyboy = PyBoy(config.ROM_PATH, window="SDL2")
    pyboy.set_emulation_speed(1)

    try:
        while pyboy.tick():
            pass
    except KeyboardInterrupt:
        pass

    # Salva
    with open(config.INIT_STATE_PATH, "wb") as f:
        pyboy.save_state(f)

    # Info
    m = pyboy.memory
    party = m[mem.PARTY_COUNT]
    level = m[mem.PARTY_LEVELS[0]] if party > 0 else 0
    map_id = m[mem.MAP_ID]
    print(f"\n[OK] Save state salvo: {config.INIT_STATE_PATH}")
    print(f"  Pokémon na party: {party}")
    print(f"  Level do primeiro: {level}")
    print(f"  Map ID: {map_id}")

    pyboy.stop()


def watch_agent(model_path: str):
    """Assiste o agente treinado jogar via SDL2."""
    print("=" * 60)
    print("   POKÉMON RED — ASSISTINDO O AGENTE JOGAR")
    print("=" * 60)
    print(f"  Modelo: {model_path}\n")

    env = PokemonRedEnv(render_mode="human")
    env.pyboy.set_emulation_speed(1)

    model = PPO.load(model_path, env=env)

    obs, _ = env.reset()
    total_reward = 0
    step_count = 0

    print("[PLAY] Agente jogando! Feche a janela ou Ctrl+C para parar.\n")

    try:
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            step_count += 1

            if step_count % 100 == 0:
                print(
                    f"  Step {step_count:>6} | "
                    f"Reward: {total_reward:>8.1f} | "
                    f"Tiles: {info.get('tiles_explored', 0):>5} | "
                    f"Badges: {info.get('badges', 0)} | "
                    f"Deaths: {info.get('deaths', 0)} | "
                    f"MapProg: {info.get('max_map_progress', 0)}"
                )

            if terminated or truncated:
                print(f"\n[EP FIM] Reward: {total_reward:.1f} | Steps: {step_count}")
                obs, _ = env.reset()
                total_reward = 0
                step_count = 0

    except KeyboardInterrupt:
        print("\n[PLAY] Interrompido.")
    except SystemExit:
        pass
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Assistir o Agente Pokémon jogar")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--create-save", action="store_true")
    args = parser.parse_args()

    if args.create_save:
        create_save_state()
        return

    model_path = args.model or find_best_model()
    if not model_path or not os.path.exists(model_path):
        print("[ERRO] Nenhum modelo treinado encontrado!")
        print("       Execute 'python train.py' primeiro.")
        return

    watch_agent(model_path)


if __name__ == "__main__":
    main()
