"""
train.py — Script de Treinamento do Agente com PPO (Stable-Baselines3).

Execução:
    python train.py                   → Treina do zero
    python train.py --resume          → Continua treinamento do último modelo salvo
    python train.py --timesteps 5000000 → Define total de timesteps
"""
import argparse
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
)

import config
from environment import PokemonRedEnv


class PokemonLogCallback(BaseCallback):
    """
    Callback customizado que loga estatísticas do jogo no console e TensorBoard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Verifica se algum episódio terminou neste step
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                self.episode_count += 1
                infos = self.locals.get("infos", [])
                if i < len(infos):
                    info = infos[i]
                    elapsed = time.time() - self.start_time
                    fps = self.num_timesteps / max(elapsed, 1)

                    # Log no console
                    print(
                        f"[EP {self.episode_count:>5}] "
                        f"Steps: {self.num_timesteps:>8,} | "
                        f"Tiles: {info.get('tiles_explored', 0):>4} | "
                        f"Mapas: {info.get('maps_explored', 0):>2} | "
                        f"Badges: {info.get('badges', 0)} | "
                        f"Level: {info.get('total_level', 0):>3} | "
                        f"Mapa: {info.get('map_name', '?'):>20} | "
                        f"FPS: {fps:>6.0f}"
                    )

                    # Log no TensorBoard
                    self.logger.record("pokemon/tiles_explored", info.get("tiles_explored", 0))
                    self.logger.record("pokemon/maps_explored", info.get("maps_explored", 0))
                    self.logger.record("pokemon/badges", info.get("badges", 0))
                    self.logger.record("pokemon/total_level", info.get("total_level", 0))
                    self.logger.record("pokemon/fps", fps)

        return True


def find_latest_model() -> str | None:
    """Encontra o modelo mais recente na pasta de modelos."""
    if not os.path.exists(config.MODEL_DIR):
        return None

    models = [
        f for f in os.listdir(config.MODEL_DIR)
        if f.startswith("pokemon_ppo_") and f.endswith(".zip")
    ]
    if not models:
        return None

    # Ordena pelo número de steps (extraído do nome do arquivo)
    models.sort(key=lambda x: int(x.replace("pokemon_ppo_", "").replace("_steps.zip", "")))
    return os.path.join(config.MODEL_DIR, models[-1])


def main():
    parser = argparse.ArgumentParser(description="Treinar o Agente Pokémon Red com RL")
    parser.add_argument("--resume", action="store_true", help="Continuar treinamento do último modelo")
    parser.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS, help="Total de timesteps")
    args = parser.parse_args()

    print("=" * 60)
    print("   POKÉMON RED — TREINAMENTO COM REINFORCEMENT LEARNING")
    print("=" * 60)
    print(f"  ROM: {config.ROM_PATH}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  LR: {config.LEARNING_RATE} | Batch: {config.BATCH_SIZE} | Gamma: {config.GAMMA}")
    print(f"  Rede: pi={config.POLICY_KWARGS['net_arch']['pi']} vf={config.POLICY_KWARGS['net_arch']['vf']}")
    print("=" * 60)

    # 1. Cria o ambiente (sem janela gráfica = velocidade máxima)
    print("\n[SETUP] Criando ambiente de treino (sem janela)...")
    env = PokemonRedEnv(render_mode="null")

    # 2. Cria ou carrega o modelo PPO
    if args.resume:
        latest = find_latest_model()
        if latest:
            print(f"[SETUP] Carregando modelo: {latest}")
            model = PPO.load(latest, env=env)
            model.learning_rate = config.LEARNING_RATE
        else:
            print("[SETUP] Nenhum modelo encontrado. Iniciando do zero.")
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=config.LEARNING_RATE,
                n_steps=config.N_STEPS,
                batch_size=config.BATCH_SIZE,
                n_epochs=config.N_EPOCHS,
                gamma=config.GAMMA,
                gae_lambda=config.GAE_LAMBDA,
                clip_range=config.CLIP_RANGE,
                ent_coef=config.ENT_COEF,
                policy_kwargs=config.POLICY_KWARGS,
                tensorboard_log=config.LOG_DIR,
                verbose=config.VERBOSE,
            )
    else:
        print("[SETUP] Criando modelo PPO do zero...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.LEARNING_RATE,
            n_steps=config.N_STEPS,
            batch_size=config.BATCH_SIZE,
            n_epochs=config.N_EPOCHS,
            gamma=config.GAMMA,
            gae_lambda=config.GAE_LAMBDA,
            clip_range=config.CLIP_RANGE,
            ent_coef=config.ENT_COEF,
            policy_kwargs=config.POLICY_KWARGS,
            tensorboard_log=config.LOG_DIR,
            verbose=config.VERBOSE,
        )

    # 3. Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=config.SAVE_FREQUENCY,
        save_path=config.MODEL_DIR,
        name_prefix="pokemon_ppo",
    )
    log_cb = PokemonLogCallback()

    # 4. TREINA!
    print("\n[TREINO] Iniciando treinamento...\n")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_cb, log_cb],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\n[TREINO] Interrompido pelo usuário.")

    # 5. Salva o modelo final
    final_path = os.path.join(config.MODEL_DIR, "pokemon_ppo_final")
    model.save(final_path)
    print(f"\n[TREINO] Modelo final salvo em: {final_path}.zip")

    env.close()
    print("[TREINO] Treinamento concluído!")


if __name__ == "__main__":
    main()
