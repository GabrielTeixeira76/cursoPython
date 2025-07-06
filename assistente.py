import speech_recognition as sr
import pyttsx3
import webbrowser
import wikipedia
import requests
import json
from datetime import datetime

class VirtualAssistant:
    def __init__(self):
        # Configuração do Text-to-Speech
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[0].id)  # Seleciona a voz padrão
        
        # Configuração do Speech-to-Text
        self.recognizer = sr.Recognizer()
        
        # Configurações do assistente
        self.name = "Assistente"
        self.wake_word = "assistente"
        
    def speak(self, text):
        """Transforma texto em fala"""
        print(f"{self.name}: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
        
    def listen(self):
        """Ouve e transforma fala em texto"""
        with sr.Microphone() as source:
            print("Ouvindo...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
            
            try:
                print("Reconhecendo...")
                query = self.recognizer.recognize_google(audio, language='pt-BR')
                print(f"Usuário: {query}")
                return query.lower()
            except Exception as e:
                print("Desculpe, não entendi. Poderia repetir?")
                return ""
    
    def process_command(self, command):
        """Processa os comandos de voz"""
        if not command:
            return
            
        if self.wake_word in command:
            # Remove a palavra de ativação do comando
            command = command.replace(self.wake_word, "").strip()
            
            # Comandos para abrir sites
            if "abrir youtube" in command:
                self.speak("Abrindo o YouTube")
                webbrowser.open("https://www.youtube.com")
                
            elif "abrir wikipedia" in command or "pesquisar na wikipedia" in command:
                self.speak("O que você gostaria de pesquisar na Wikipedia?")
                search_query = self.listen()
                if search_query:
                    try:
                        result = wikipedia.summary(search_query, sentences=2)
                        self.speak("De acordo com a Wikipedia: " + result)
                    except:
                        self.speak("Não encontrei informações sobre isso na Wikipedia.")
            
            # Comando para localizar farmácias
            elif "farmacia mais proxima" in command or "localizar farmacia" in command:
                self.locate_pharmacy()
                
            # Comandos de hora e data
            elif "que horas são" in command:
                now = datetime.now()
                current_time = now.strftime("%H:%M")
                self.speak(f"Agora são {current_time}")
                
            elif "que dia é hoje" in command:
                today = datetime.now().strftime("%d de %B de %Y")
                self.speak(f"Hoje é {today}")
                
            else:
                self.speak("Desculpe, não entendi o comando. Posso ajudar com algo mais?")
    
    def locate_pharmacy(self):
        """Localiza a farmácia mais próxima usando API do Google Maps"""
        self.speak("Vou localizar a farmácia mais próxima.")
        
        # Em um sistema real, você usaria a API do Google Places
        # Esta é uma implementação simulada
        try:
            # Simulando uma chamada à API
            # Na prática, você precisaria de uma chave de API e fazer uma requisição real
            pharmacy_info = {
                "name": "Drogaria São Paulo",
                "address": "Rua Exemplo, 123 - Centro",
                "distance": "500 metros"
            }
            
            self.speak(f"A farmácia mais próxima é a {pharmacy_info['name']}, localizada a {pharmacy_info['distance']} de distância, no endereço {pharmacy_info['address']}")
            self.speak("Deseja que eu mostre no mapa?")
            
            response = self.listen()
            if "sim" in response:
                webbrowser.open("https://www.google.com/maps/search/farmácia+próxima")
        except:
            self.speak("Não consegui acessar a localização no momento. Você pode tentar abrir o mapa manualmente.")
    
    def run(self):
        """Inicia o assistente virtual"""
        self.speak(f"Olá! Eu sou seu {self.name}. Diga '{self.wake_word}' seguido de um comando para interagir comigo.")
        
        while True:
            try:
                command = self.listen()
                self.process_command(command)
            except KeyboardInterrupt:
                self.speak("Encerrando o assistente. Até mais!")
                break

if __name__ == "__main__":
    assistant = VirtualAssistant()
    assistant.run()