# Pokémon Red AI — Reinforcement Learning

Agente autônomo que aprende a jogar Pokémon Red usando **Aprendizado por Reforço (PPO)** via Stable-Baselines3 + PyBoy.

## Requisitos

- Python 3.10+
- ROM do Pokémon Red (.gb)

## Instalação

```bash
python -m venv venv
.\venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Como Usar

### 1. Criar Save State (Recomendado)
Jogue manualmente até escolher seu Pokémon inicial e feche a janela:
```bash
python play.py --create-save
```

### 2. Treinar o Agente
```bash
python train.py                       # Treina do zero (1M steps)
python train.py --timesteps 5000000   # Treinar por mais tempo
python train.py --resume              # Continuar treino anterior
```

### 3. Assistir o Agente Jogar
```bash
python play.py                        # Usa o melhor modelo disponível
python play.py --model models/pokemon_ppo_final.zip
```

### 4. Monitorar com TensorBoard
```bash
tensorboard --logdir logs/
```

## Arquitetura

| Arquivo | Responsabilidade |
|---|---|
| `memory_map.py` | Endereços de memória RAM do Pokémon Red |
| `rewards.py` | Sistema de recompensas (exploração, badges, XP, HP) |
| `environment.py` | Gymnasium Environment envolvendo o PyBoy |
| `config.py` | Hiperparâmetros centralizados |
| `train.py` | Script de treinamento com PPO |
| `play.py` | Assistir o agente treinado jogar |

## Ajustes

Todos os hiperparâmetros estão em `config.py`. Os mais importantes:

- `TOTAL_TIMESTEPS`: Quanto tempo treinar (mais = melhor, mas mais lento)
- `REWARD_NEW_TILE`: Recompensa por explorar tiles novos
- `REWARD_BADGE`: Recompensa por ganhar badges
- `FRAMES_PER_ACTION`: Frames por ação (afeta velocidade de decisão)
