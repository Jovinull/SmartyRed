from pyboy import PyBoy
from pyboy.utils import WindowEvent
import requests
import json
import time

class AgentePokemon:
    def __init__(self, rom_path, url_ia, modelo_ia):
        # 1. Configurações da IA (Seu LM Studio)
        self.url_ia = url_ia
        self.modelo_ia = modelo_ia
        
        # 2. Inicia o Emulador PyBoy
        print(f"[DEBUG] Tentando abrir ROM: {rom_path}")
        try:
            # Usamos SDL2 para ver a janela. Se o crash continuar, mude para "null" para testar apenas o código.
            self.pyboy = PyBoy(rom_path, window="SDL2") 
            print("[DEBUG] Objeto PyBoy criado.")
            self.pyboy.set_emulation_speed(1)
            print("[DEBUG] Velocidade definida.")
        except Exception as e:
            print(f"[ERRO] Falha na inicialização do PyBoy: {e}")
            raise e
        
        # Mapeamento de texto para os eventos do PyBoy
        self.botoes = {
            "up": (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
            "down": (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
            "left": (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
            "right": (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
            "a": (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
            "b": (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
            "start": (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START)
        }

    def ler_estado_do_jogo(self):
        """
        Aqui é onde a mágica acontece no futuro.
        A IA baseada em texto não 'vê' a tela. Nós precisamos ler a memória RAM do jogo.
        Por enquanto, vamos simular um estado genérico para testar o motor.
        """
        # Exemplo do que você programaria lendo a memória do PyBoy:
        # hp = self.pyboy.get_memory_value(0x0202423C)
        # mapa_atual = self.pyboy.get_memory_value(0x02031DBC)
        
        contexto = """
        Você está jogando Pokémon FireRed. 
        Seu objetivo é explorar o mapa, conversar com NPCs e vencer batalhas.
        Opções de botões: 'up', 'down', 'left', 'right', 'a', 'b', 'start'.
        """
        return contexto

    def pensar(self, contexto):
        """ Envia o texto para o Qwen 3B no LM Studio e exige um JSON """
        prompt_sistema = """
        Você é uma Inteligência Artificial jogando Pokémon no Gameboy.
        Responda APENAS com um objeto JSON válido contendo a chave "botao".
        NENHUM TEXTO ADICIONAL.
        Exemplo: {"botao": "a"}
        """
        
        payload = {
            "model": self.modelo_ia,
            "messages": [
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": f"O que você aperta agora?\nContexto: {contexto}"}
            ],
            "temperature": 0.7 # Um pouco de criatividade para ele tentar andar por caminhos diferentes
        }

        try:
            resposta = requests.post(self.url_ia, json=payload, timeout=30)
            conteudo = resposta.json()['choices'][0]['message']['content']
            
            # Limpa qualquer texto extra que a IA possa gerar fora do JSON
            inicio_json = conteudo.find('{')
            fim_json = conteudo.rfind('}') + 1
            json_limpo = conteudo[inicio_json:fim_json]
            
            return json.loads(json_limpo)
            
        except Exception as e:
            print(f"[Erro no Cérebro]: {e}")
            return {"botao": "a"} # Ação padrão de segurança para não travar o jogo

    def apertar_botao(self, botao_str):
        """ Simula o dedo apertando e soltando o botão no Gameboy """
        if botao_str in self.botoes:
            evento_press, evento_release = self.botoes[botao_str]
            
            # Aperta o botão
            self.pyboy.send_input(evento_press)
            # Avança alguns frames de jogo segurando o botão
            for _ in range(5): 
                self.pyboy.tick()
                
            # Solta o botão
            self.pyboy.send_input(evento_release)
            print(f"IA Apertou: {botao_str.upper()}")

    def _chamar_ia_async(self, estado_atual):
        """ Função que roda em uma thread separada para não travar o jogo """
        try:
            decisao = self.pensar(estado_atual)
            botao_escolhido = decisao.get("botao", "a").lower()
            self.proxima_acao = botao_escolhido
        except Exception as e:
            print(f"[Erro Thread IA]: {e}")
            self.proxima_acao = "a"
        finally:
            self.ia_pensando = False

    def iniciar_loop(self):
        """ O coração do agente. Um ciclo infinito de Ler -> Pensar -> Agir """
        print("\n[INFO] IA assumiu o controle.")
        print("[SISTEMA] Pressione Ctrl+C no terminal para parar.\n")
        
        self.ia_pensando = False
        self.proxima_acao = None
        frames_desde_ultima_acao = 0
        import threading

        try:
            while True:
                # 1. O Jogo nunca para! Sempre avançamos o frame.
                if not self.pyboy.tick():
                    print("\n[AVISO] Janela fechada ou sinal de parada.")
                    break
                
                frames_desde_ultima_acao += 1

                # 2. Se a IA não estiver pensando e já passou um tempinho (ex. 60 frames = 1s), pede nova decisão
                if not self.ia_pensando and frames_desde_ultima_acao > 60:
                    estado_atual = self.ler_estado_do_jogo()
                    
                    self.ia_pensando = True
                    # Dispara a requisição em segundo plano
                    thread_ia = threading.Thread(target=self._chamar_ia_async, args=(estado_atual,))
                    thread_ia.daemon = True # Para a thread morrer se o programa fechar
                    thread_ia.start()
                
                # 3. Se a thread terminou e temos uma ação na fila, executamos!
                if self.proxima_acao is not None:
                    print(f"IA decidiu: {self.proxima_acao.upper()}   ")
                    self.apertar_botao(self.proxima_acao)
                    self.proxima_acao = None
                    frames_desde_ultima_acao = 0 # Reseta o contador
                    
        except KeyboardInterrupt:
            print("\n[SISTEMA] Desligando o Gameboy...")
        except Exception as e:
            print(f"\n[ERRO CRÍTICO]: {e}")
        finally:
            try:
                self.pyboy.stop()
            except Exception:
                pass

# ==========================================
# CONFIGURAÇÕES E INICIALIZAÇÃO
# ==========================================
if __name__ == "__main__":
    ROM = "Pokemon_FireRed.gb"
    URL_LM_STUDIO = "http://127.0.0.1:1234/v1/chat/completions"
    MODELO = "qwen2.5-coder-3b-instruct"

    # Verifica se a ROM existe antes de começar
    import os
    if not os.path.exists(ROM):
        print(f"ERRO: Arquivo '{ROM}' não encontrado na pasta!")
    else:
        try:
            agente = AgentePokemon(ROM, URL_LM_STUDIO, MODELO)
            agente.iniciar_loop()
        except Exception as e:
            print(f"Não foi possível iniciar o agente: {e}")