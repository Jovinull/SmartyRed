"""
train.py — Script de Treinamento com PPO (Stable-Baselines3).

Adaptado do PokemonRedExperiments V2.
Suporta continuação automática do último checkpoint.

Uso:
    python train.py                       → Treina (continua se existir checkpoint)
    python train.py --fresh               → Força treino do zero
    python train.py --timesteps 5000000   → Define total de timesteps
"""
import argparse
import os
import sys
import glob
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
    CallbackList,
)

import config
from environment import PokemonRedEnv


class PokemonLogCallback(BaseCallback):
    """Loga estatísticas do jogo no console e TensorBoard a cada episódio."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.start_time = time.time()

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                self.episode_count += 1
                infos = self.locals.get("infos", [])
                if i < len(infos):
                    info = infos[i]
                    elapsed = time.time() - self.start_time
                    fps = self.num_timesteps / max(elapsed, 1)

                    print(
                        f"[EP {self.episode_count:>4}] "
                        f"Steps: {self.num_timesteps:>9,} | "
                        f"Tiles: {info.get('tiles_explored', 0):>5} | "
                        f"Badges: {info.get('badges', 0)} | "
                        f"Level: {info.get('total_level', 0):>3} | "
                        f"Deaths: {info.get('deaths', 0):>2} | "
                        f"HP: {info.get('hp', 0):.2f} | "
                        f"Reward: {info.get('total_reward', 0):>8.1f} | "
                        f"MapProg: {info.get('max_map_progress', 0):>2} | "
                        f"FPS: {fps:>5.0f}"
                    )

                    # TensorBoard
                    self.logger.record("pokemon/tiles_explored", info.get("tiles_explored", 0))
                    self.logger.record("pokemon/badges", info.get("badges", 0))
                    self.logger.record("pokemon/total_level", info.get("total_level", 0))
                    self.logger.record("pokemon/deaths", info.get("deaths", 0))
                    self.logger.record("pokemon/hp", info.get("hp", 0))
                    self.logger.record("pokemon/total_reward", info.get("total_reward", 0))
                    self.logger.record("pokemon/max_map_progress", info.get("max_map_progress", 0))
                    self.logger.record("pokemon/event_reward", info.get("event_reward", 0))
                    self.logger.record("pokemon/healing_reward", info.get("healing_reward", 0))
                    self.logger.record("pokemon/fps", fps)
        return True


def find_latest_checkpoint() -> str | None:
    """Encontra o checkpoint mais recente na pasta de modelos."""
    if not os.path.exists(config.MODEL_DIR):
        return None
    zips = glob.glob(os.path.join(config.MODEL_DIR, "pokemon_ppo_*.zip"))
    if not zips:
        return None
    # Ordena pelo tempo de modificação (o mais novo por último)
    zips.sort(key=os.path.getmtime)
    return zips[-1]


def main():
    parser = argparse.ArgumentParser(description="Treinar Agente Pokémon Red com RL")
    parser.add_argument("--fresh", action="store_true", help="Forçar treino do zero")
    parser.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS)
    args = parser.parse_args()

    print("=" * 70)
    print("   POKÉMON RED — TREINAMENTO COM REINFORCEMENT LEARNING (PPO)")
    print("   Adaptado do PokemonRedExperiments V2")
    print("=" * 70)
    print(f"  ROM: {config.ROM_PATH}")
    print(f"  Init state: {config.INIT_STATE_PATH}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Episódio: {config.MAX_STEPS_PER_EPISODE:,} steps ({config.MAX_STEPS_PER_EPISODE * config.ACTION_FREQ / 60:.0f}s de jogo)")
    print(f"  PPO: n_steps={config.N_STEPS} batch={config.BATCH_SIZE} epochs={config.N_EPOCHS}")
    print(f"  gamma={config.GAMMA} ent_coef={config.ENT_COEF} lr={config.LEARNING_RATE}")
    print(f"  Reward: scale={config.REWARD_SCALE} explore_w={config.EXPLORE_WEIGHT}")
    print("=" * 70)

    # 1. Cria o ambiente (sem janela = velocidade máxima)
    print("\n[SETUP] Criando ambiente de treino (headless)...")
    env = PokemonRedEnv(render_mode="null")

    # 2. Cria ou carrega o modelo
    latest = None if args.fresh else find_latest_checkpoint()

    if latest:
        print(f"[SETUP] Continuando do checkpoint: {latest}")
        model = PPO.load(latest, env=env)
        # Restaura parâmetros que podem ter mudado no config
        model.n_steps = config.N_STEPS
        model.n_envs = config.NUM_ENVS
        model.rollout_buffer.buffer_size = config.N_STEPS
        model.rollout_buffer.n_envs = config.NUM_ENVS
        model.rollout_buffer.reset()
    else:
        print("[SETUP] Criando modelo PPO do zero...")
        model = PPO(
            "MultiInputPolicy",  # Usa Dict observation!
            env,
            learning_rate=config.LEARNING_RATE,
            n_steps=config.N_STEPS,
            batch_size=config.BATCH_SIZE,
            n_epochs=config.N_EPOCHS,
            gamma=config.GAMMA,
            gae_lambda=config.GAE_LAMBDA,
            clip_range=config.CLIP_RANGE,
            ent_coef=config.ENT_COEF,
            tensorboard_log=config.LOG_DIR,
            verbose=config.VERBOSE,
        )

    print(f"\n[SETUP] Política: {type(model.policy).__name__}")
    print(f"[SETUP] Parâmetros: {sum(p.numel() for p in model.policy.parameters()):,}")

    # 3. Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=config.SAVE_FREQUENCY,
        save_path=config.MODEL_DIR,
        name_prefix="pokemon_ppo",
    )
    log_cb = PokemonLogCallback()

    # 4. TREINA!
    print("\n[TREINO] Iniciando... (Ctrl+C para parar e salvar)\n")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList([checkpoint_cb, log_cb]),
            progress_bar=True,
            tb_log_name="pokemon_ppo",
        )
    except KeyboardInterrupt:
        print("\n\n[TREINO] Interrompido pelo usuário.")

    # 5. Salva modelo final
    final_path = os.path.join(config.MODEL_DIR, "pokemon_ppo_final")
    model.save(final_path)
    print(f"\n[TREINO] Modelo salvo em: {final_path}.zip")
    print("[TREINO] Para continuar, basta rodar: python train.py")

    env.close()


if __name__ == "__main__":
    main()
