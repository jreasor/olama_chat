import os
import sys
import signal
import gradio as gr
import threading
import webview
import asyncio
from ollama_chatbot import OllamaChatbot

class App:
    def __init__(self, chatbot=None):
        self.chatbot = chatbot
        self.loop = asyncio.new_event_loop()
        self.server_thread = threading.Thread(target=self.run_chatbot, daemon=True)
        
    def run_chatbot(self):
        """Run the chatbot in an asyncio event loop"""
        asyncio.set_event_loop(self.loop)
        if self.chatbot:
            self.loop.run_until_complete(self.chatbot.launch_ui())
        
    def cleanup_and_exit(self):
        print("Exit called. Saving chat history...")
        if self.chatbot:  
            self.chatbot.save_chat_history()
        sys.exit(0)
        

    def signal_handler(self, sig, frame):
        self.cleanup_and_exit()
        
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    chatbot = OllamaChatbot(inbrowser=False)
    app = App(chatbot)
    signal.signal(signal.SIGINT, app.signal_handler)
    signal.signal(signal.SIGTERM, app.signal_handler)  
    
    app.server_thread.start()
    
    try:
        webview.create_window(
            title=chatbot.title,
            url=f"http://{chatbot.gradio_server_name}:{chatbot.gradio_server_port}",
            text_select=True,
            zoomable=True,
            draggable=True,
            vibrancy=True
        )
        webview.start()
    except KeyboardInterrupt:
        app.signal_handler(None, None)
    finally:
        app.cleanup_and_exit()

if __name__ == "__main__":
    main()
