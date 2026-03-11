# Pokémon Red AI — Reinforcement Learning

Agente autônomo que **aprende sozinho** a jogar Pokémon Red usando Aprendizado por Reforço (PPO).

Adaptado do [PokemonRedExperiments](https://github.com/pwhiddy/PokemonRedExperiments) V2 por Peter Whidden.

## Arquitetura

O agente "enxerga" o jogo através de:
- **3 frames empilhados** da tela (memória visual)
- **Fração de HP** de toda a party
- **Mapa de exploração** local (onde já andou)
- **Badges** conquistadas (vetor binário)
- **~1700 event flags** do jogo (progresso na história!)
- **Últimas 3 ações** tomadas

E é recompensado por:
- ✅ Ativar event flags (falar com NPCs, pegar itens, derrotar treinadores)
- ✅ Curar no Pokémon Center
- ✅ Conquistar badges
- ✅ Explorar tiles novos
- ❌ Ficar andando em círculos (penalidade)

## Instalação

```bash
python -m venv venv
.\venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Como Usar

### 1. (Opcional) Criar Save State personalizado
```bash
python play.py --create-save
```

### 2. Treinar o Agente
```bash
python train.py                         # Treina (continua automaticamente se já tiver checkpoint)
python train.py --fresh                 # Força treino do zero
python train.py --timesteps 20000000    # Treinar por mais tempo
```

> **Dica**: O treino **continua automaticamente** do último checkpoint ao rodar `python train.py` de novo. Não precisa do flag `--resume`!

### 3. Assistir o Agente Jogar
```bash
python play.py
```

### 4. Monitorar com TensorBoard
```bash
tensorboard --logdir logs/
```

## Estrutura do Projeto

| Arquivo | Responsabilidade |
|---|---|
| `environment.py` | Gymnasium Env com observação multimodal (Dict) |
| `memory_map.py` | Endereços RAM do Pokémon Red (party, badges, events) |
| `global_map.py` | Conversão de coordenadas locais → globais |
| `config.py` | Hiperparâmetros centralizados |
| `train.py` | Treinamento PPO com checkpoint automático |
| `play.py` | Assistir agente / Criar save state |
| `events.json` | Nomes das event flags do jogo |
| `map_data.json` | Dados de posição de cada mapa |
| `init.state` | Save state inicial do emulador |

## Hardware Recomendado

- **Mínimo**: i5 + 16GB RAM (treino single-CPU)
- **Ideal**: GPU NVIDIA + 64 CPUs (como no projeto referência)

Com um i5 13ª gen, espere resultados visíveis após ~2-5M steps (algumas horas).
