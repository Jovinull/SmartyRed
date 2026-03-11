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

### 1. Criar Save State Personalizado (Altamente Recomendado)
```bash
python play.py --create-save
```

**Por que isso é essencial?**
Sem dúvida nenhuma: o melhor é iniciar através de um Save State logo APÓS pegar o primeiro Pokémon (e idealmente após pegar a Pokédex).

Aqui estão os 3 motivos técnicos de por que isso é o padrão em projetos de alto nível:

1. **O "Deserto de Recompensas"**: Se você começar do zero absoluto (no quarto do jogador), a IA levaria milhares de passos apenas para descer a escada, sair da casa, falar com o Carvalho e passar por todos os diálogos de introdução. Como ela ainda não tem um Pokémon, ela não ganha XP, não perde HP e não tem progresso de batalha. Para a IA, esse início é um "vazio" sem feedback, o que torna o aprendizado inicial extremamente lento e frustrante.
2. **Eficiência de Hardware**: Rodando no seu i5, você quer que cada segundo de processamento ensine algo útil. Treinar a IA para "atravessar diálogos de 5 minutos" é um desperdício. Colocando ela no início da Rota 1, cada passo que ela dá pode resultar em uma batalha, um ganho de XP ou a descoberta de um novo mapa. O "aprendizado por segundo" é muito maior.
3. **Foco na Gameplay Real**: O objetivo é que a IA aprenda a lutar, gerenciar HP e navegar pelo mundo. A introdução do jogo é quase um "filme interativo" com pouca liberdade. Começar com o Pokémon em mãos foca o cérebro da rede neural no que realmente importa: a jornada Pokémon.

**Recomendação de como fazer:**
1. Execute o comando `.\venv\Scripts\python play.py --create-save`.
2. Jogue manualmente, escolha seu inicial (Charmander/Squirtle/Bulbasaur), e pegue a Pokédex com o Carvalho para liberar as rotas.
3. Vá até a saída de Pallet Town ou já na Rota 1.
4. Feche a janela. Pronto! Toda vez que a IA resetar, já nascerá ali.

### 2. Treinar o Agente
```bash
python train.py                         # Treina (continua automaticamente se já tiver checkpoint)
python train.py --fresh                 # Força treino do zero
python train.py --timesteps 20000000    # Treinar por mais tempo
python train.py --num-envs 8            # Define a quantidade de processos paralelos
```

> **Dica**: O treino **continua automaticamente** do último checkpoint ao rodar `python train.py` de novo. Não precisa do flag `--resume`!

> **Processamento Paralelo**: Assim como o projeto base V2, o seu agente roda em ambientes (emuladores) separados simultâneos em Subprocessos (e não dependendo de threads limitadas pelo Python). Você pode configurar a quantidade de instâncias passando o argumento `--num-envs N` (ou editando `NUM_ENVS` no `config.py`) para consumir mais poder do se PC (ex: i5 de 13ª geração com múltiplos núcleos usa 8 processos de forma brilhante).

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

- **Mínimo**: CPU moderna (ex: i5 13ª gen) + 16GB RAM rodando **8 processos paralelos**. A arquitetura do `SubprocVecEnv` distribuirá o peso de cada emulador pelos seus núcleos lógicos.
- **Ideal**: GPU NVIDIA pesada + 64 processos paralelos (como no projeto de referência em High-End servers).

Com seu hardware rodando os 8 processos configurados no `config.py`, multiplicamos os ensinamentos adquiridos a cada vez que a rede neural é atualizada, trazendo resultados em muito menos tempo (algumas horas em vez de dias).
