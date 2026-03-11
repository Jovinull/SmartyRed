"""
train.py — Script de Treinamento Paralelo com PPO (Stable-Baselines3).

Usa SubprocVecEnv para rodar múltiplos emuladores em paralelo,
igual ao PokemonRedExperiments.

Uso:
    python train.py                       → Treina (continua se existir checkpoint)
    python train.py --fresh               → Força treino do zero
    python train.py --timesteps 5000000   → Define total de timesteps
    python train.py --num-envs 4          → Usa 4 processos paralelos
"""
import argparse
import os
import glob
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
    CallbackList,
)

import config


def make_env(rank, seed=0):
    """
    Cria uma função factory para o SubprocVecEnv.
    Cada processo vai chamar essa função e receber seu próprio PyBoy.
    """
    def _init():
        # Import dentro da função para que cada processo tenha seu próprio módulo
        from environment import PokemonRedEnv
        env = PokemonRedEnv(render_mode="null")
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


class PokemonLogCallback(BaseCallback):
    """Loga estatísticas do jogo no console e TensorBoard a cada episódio."""

    def __init__(self, num_envs, verbose=0):
        super().__init__(verbose)
        self.num_envs = num_envs
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
                        f"Env {i:>2}/{self.num_envs} | "
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
    zips.sort(key=os.path.getmtime)
    return zips[-1]


def main():
    parser = argparse.ArgumentParser(description="Treinar Agente Pokémon Red com RL")
    parser.add_argument("--fresh", action="store_true", help="Forçar treino do zero")
    parser.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS)
    parser.add_argument("--num-envs", type=int, default=config.NUM_ENVS,
                        help=f"Processos paralelos (default: {config.NUM_ENVS})")
    args = parser.parse_args()

    num_envs = args.num_envs

    # n_steps por env — total de samples por rollout = n_steps * num_envs
    n_steps = config.N_STEPS
    total_samples_per_rollout = n_steps * num_envs

    print("=" * 70)
    print("   POKÉMON RED — TREINAMENTO PARALELO COM RL (PPO)")
    print("=" * 70)
    print(f"  ROM: {config.ROM_PATH}")
    print(f"  Init state: {config.INIT_STATE_PATH}")
    print(f"  Timesteps alvo: {args.timesteps:,}")
    print(f"  Processos paralelos: {num_envs} (= {num_envs} Game Boys simultâneos)")
    print(f"  Episódio: {config.MAX_STEPS_PER_EPISODE:,} steps por env")
    print(f"  PPO: n_steps={n_steps} × {num_envs} envs = {total_samples_per_rollout:,} samples/rollout")
    print(f"  batch={config.BATCH_SIZE} epochs={config.N_EPOCHS}")
    print(f"  gamma={config.GAMMA} ent_coef={config.ENT_COEF} lr={config.LEARNING_RATE}")
    print(f"  Reward: scale={config.REWARD_SCALE} explore_w={config.EXPLORE_WEIGHT}")
    print(f"  RAM estimada: ~{num_envs * 150}MB para emuladores")
    print("=" * 70)

    # 1. Cria ambientes paralelos
    print(f"\n[SETUP] Criando {num_envs} ambientes paralelos (headless)...")
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    print(f"[SETUP] {num_envs} Game Boys rodando em processos separados!")

    # 2. Cria ou carrega o modelo
    latest = None if args.fresh else find_latest_checkpoint()

    if latest:
        print(f"[SETUP] Continuando do checkpoint: {latest}")
        model = PPO.load(latest, env=env)
        # Atualiza parâmetros para o novo número de envs
        model.n_steps = n_steps
        model.n_envs = num_envs
        model.rollout_buffer.buffer_size = n_steps
        model.rollout_buffer.n_envs = num_envs
        model.rollout_buffer.reset()
    else:
        print("[SETUP] Criando modelo PPO do zero...")
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=config.LEARNING_RATE,
            n_steps=n_steps,
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
    print(f"[SETUP] Parâmetros da rede: {sum(p.numel() for p in model.policy.parameters()):,}")

    # 3. Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(config.SAVE_FREQUENCY // num_envs, 1),
        save_path=config.MODEL_DIR,
        name_prefix="pokemon_ppo",
    )
    log_cb = PokemonLogCallback(num_envs=num_envs)

    # 4. TREINA!
    print(f"\n[TREINO] Iniciando com {num_envs} processos... (Ctrl+C para parar e salvar)\n")
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
