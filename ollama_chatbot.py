import os
import signal
import sys
import subprocess
import json
import yaml
import time
import gradio as gr
import ollama
import faiss
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

class OllamaChatbot:
    def __init__(self,
                 title="Olama Chatbot",
                 gradio_server_name="localhost",
                 gradio_server_port=7860,
                 inbrowser = True,
                 client=None,
                 models=None,
                 model=None,
                 default_config_file="config/default_config.yaml",
                 config_file="config/config.yaml",
                 config=None,
                 ollama_mode="Chat",
                 chat_initiated=False,
                 chat_save_folder="chat_history",
                 current_save_file=None,
                 chat_history=None,
                 chat_list=None,
                 summary="",
                 session_timestamp=None,
                 index=None,
                 doc_path=None,
                 top_doc=3,
                 retrieved_docs=None,
                 layout=None,
                 css_file="config/styles.css",
                 styles=""):
        self.title = title
        self.gradio_server_name = gradio_server_name
        self.gradio_server_port = gradio_server_port
        self.inbrowser = inbrowser
        self.client = client
        self.models = models
        self.model = model
        self.default_config_file = default_config_file
        self.config_file = config_file
        self.config = config
        self.ollama_mode = ollama_mode
        self.chat_initiated = chat_initiated
        self.chat_save_folder = chat_save_folder
        self.current_save_file = current_save_file
        self.chat_history = chat_history
        self.chat_list = chat_list
        self.summary = summary
        self.session_timestamp = session_timestamp
        self.index = index
        self.doc_path = doc_path
        self.top_doc = top_doc
        self.retrieved_docs = retrieved_docs
        self.layout = layout
        self.css_file = css_file
        self.styles = styles
        
        self.create_folder(self.chat_save_folder)
        
        if not self.config:
            self.load_config()
        
        if not self.client:
            default_host = "localhost"
            default_port = "11434"
            try:
                if self.config:
                    self.client = ollama.Client(host=f"{self.config['host']}:{self.config['port']}")
                else:
                    self.client = ollama.Client(host=f"{default_host}:{default_port}")
            except ollama.RequestError as e:
                message = f"Error loading ollama: {e}"
                print(message, file=sys.stderr)
                gr.Error(message=message)
                sys.exit(1)
            except Exception as e:
                message = f"Error loading ollama: {e}"
                print(message, file=sys.stderr)
                gr.Error(message=message)
                sys.exit(1)
                
        if not self.models:
            self.get_available_models()
            
        if not self.chat_list:
            self.load_chat_history()
            
        if not self.layout:
            if self.config and 'layout' in self.config:
                self.layout = self.config['layout']
                
        if not self.doc_path:
            if self.config and 'documents_path' in self.config:
                self.doc_path = self.config['documents_path']
                
        if not len(self.styles):
            content = ""
            try:
                
                if os.path.exists(self.css_file):
                    with open(self.css_file, "r") as f:
                        content = f.read()
                
            except FileNotFoundError:    
                message = "Unable to read css file '{self.css_file}'"
                print(message, file=sys.stderr)
                gr.Error(message=message)
                
            self.styles = f"<style>\n{content}\n</style>"
        
    class ChatList:
        def __init__(self, summary, history, filename):
            self.filename = filename
            self.summary = summary
            self.history = history

    def signal_handler(self, sig, frame):
        self.save_chat_history()
        sys.exit(0)
        
    def read_yaml(self, file_path):
        """
        Reads a YAML file and returns its contents as a dictionary.
        
        Args:
            file_path (str): The path to the YAML file.
            
        Returns:
            dict: The parsed YAML content.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """
        data = None
        with open(file_path, 'r') as file:
            try:
                data = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                raise Exception(f"Error reading YAML file: {exc}") from None
        return data
            
    def write_yaml(self, file_path, data, overwrite=True):
        """
        Writes or updates a dictionary to a YAML file.
        
        If the file exists and 'overwrite' is False, new key-value pairs will be added,
        merging with existing data. If 'overwrite' is True (default), the entire content
        of the file will be replaced with the provided data.
        
        Args:
            file_path (str): The path to the YAML file.
            data (dict): The dictionary to write into the YAML file.
            overwrite (bool, optional): Whether to replace existing keys or merge them. Defaults to True.
            
        Raises:
            Exception: If there is an error writing to the file.
        """
        
        write_data = data
        
        if os.path.exists(file_path):   
            with open(file_path, 'r') as file:
                try:
                    existing_data = yaml.safe_load(file)
                except FileNotFoundError:
                    existing_data = {}
            
            if not overwrite:
               write_data = {**existing_data, **data}
        
        with open(file_path, 'w') as file:
            yaml.safe_dump(write_data, file)

    def load_config(self):
        default_host = "localhost"
        default_port = "11434"
        default_layout = "panel"
        default_embedding_model = "msmarco-bert-base-dot-v5:768"
        default_json = {"host": default_host,
                        "port": default_port,
                        "layout": default_layout,
                        "embedding_model": default_embedding_model
                        }
        
        try:
            config = None
            
            if os.path.exists(self.config_file):
                config = self.read_yaml(self.config_file)
                
            elif os.path.exists(self.default_config_file):
                config = self.read_yaml(self.default_config_file)

            if config:                    
                if "host" not in config:
                    config['host'] = default_host
                    
                if "port" not in config:
                    config['port'] = default_port
                    
                if "layout" not in config:
                    config['layout'] = default_layout
                    
                if "embedding_model" not in config:
                    config['embedding_model'] = default_embedding_model
                
                self.config = config
            else:
                self.config = default_json
        except FileNotFoundError:
            self.config = default_json
        except json.JSONDecodeError:
            self.config = default_json
        
    def save_config(self, property_name, data):
        try:
            if not isinstance(data, str):
                raise ValueError("Data must be a string")
            
            if self.config:
                self.config[property_name] = data
            else:
                self.config = {}
                self.config[property_name] = data
                
            self.write_yaml(file_path=self.config_file, data=self.config)

        except Exception as e:
            raise ValueError(f"Error saving configuration: {e}")

    def get_available_models(self):
        try:
            model = "mistral"
            response = ollama.list()
            
            if response and hasattr(response, 'models'):
                
                models = []
                
                for model in response.models:
                    models.append(model.model)
                
                self.models = sorted(models)

            else:
                ollama.pull(model)
                self.models = [model]
        except ollama.RequestError as e:
            message = f"Error returning available models from Ollama: {e}"
            print(message, file=sys.stderr)
            gr.Error(message=message)
            self.models = []
        except Exception as e:
            message = f"Error returning available models from Ollama: {e}"
            print(message, file=sys.stderr)
            gr.Error(message=message)
            self.models = []
            
        if not self.models or self.models == []:
            print("No models found.\nDownload models using the following command:\n\tollama pull <model_name>")
            sys.exit(1)
        
    def update_default_model(self, selected_model):
        self.model = selected_model
        self.save_config(property_name="default_model", data=selected_model)
        
    def create_folder(self, folder):
        try:
            os.makedirs(folder, exist_ok=True)
        except PermissionError:
            print("Permission denied. Unable to create folder '{folder}'.", file=sys.stderr)
            print("Unable to save chat history for this session.", file=sys.stderr)
        except OSError as e:
            print(f"An error occurred while creating the folder: {e}", file=sys.stderr)
            print("Unable to save chat history for this session.", file=sys.stderr)
            
    def pull_model(self, model):
        ollama.pull(model)
            
    def set_summary(self, message):
        model = "mistral"
        if not self.models or model not in self.models:
            self.pull_model(model)
        if self.summary == "" and self.client and self.model:
            response = self.client.generate(
                model=model,
                prompt=f"I plan to save this chat to a database. Could you please summarize the following phrase into a brief title I could use to tag this chat. Please only provide one brief response because this will be used as the summary. Here is the phrase we need to summarize: {message}"
            )
        
            if "response" in response:
                self.summary = response['response'].lower().replace('"', '')

    def load_chat_history(self):
        chat_history = []
        files = [f for f in os.listdir(self.chat_save_folder) if os.path.isfile(os.path.join(self.chat_save_folder, f))]
        
        try:
            for file in files:
                path = f"{self.chat_save_folder}/{file}"
                with open(path, 'r') as f:
                    content = f.read()
                    
                    if len(content) != 0 and content != "{}":
                        json_content = json.loads(content)
                        summary = json_content['summary']
                        chat_history = json_content['history']
                        if self.chat_list:
                            self.chat_list.append(self.ChatList(filename=file, summary=summary, history=chat_history))
                        else:
                            self.chat_list = [self.ChatList(filename=file, summary=summary, history=chat_history)]
        except FileNotFoundError:
            print(f"Unable to find chat history file at path: {self.chat_save_folder}", file=sys.stderr)
        except json.JSONDecodeError:
            print("Error reading JSON data from chat history file", file=sys.stderr)
        
    def save_chat_history(self):
        if not self.chat_initiated or self.chat_history== []:
            return
        if not self.session_timestamp:
            self.session_timestamp = int(time.time())
            
        if not self.current_save_file:
            basename = f"chat_history_{self.session_timestamp}"
            self.current_save_file = f"{basename}.json"
        path = f"{self.chat_save_folder}/{self.current_save_file}"
        
        if self.summary == "":
            basename = self.current_save_file.split('.')[0]
            json_content = {"summary": basename, "history": self.chat_history}
        else:
            json_content = {"summary": self.summary, "history": self.chat_history}
        
        with open(path, 'w') as f:
            json.dump(json_content, f)
            
    def update_chat_history(self, history, file):
        self.chat_history = history
        self.current_save_file = file
        return history
    
    def update_progress_bar(self, value):
        if value is not None:
            return f"""
                <div class="progress-bar-container">
                    <div class="progress-bar-background"></div>
                    <div class="progress-bar-foreground" style="width: {value}%;"></div>
                    <div class="progress-bar-text">{value}%</div>
                </div>
            """
    
    def select_folder(self):
        """Opens Finder's folder picker using AppleScript and returns the selected path."""
        script = '''
        tell application "System Events"
            activate
            set theFolder to choose folder with prompt "Select a folder:"
            return POSIX path of theFolder
        end tell
        '''

        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        self.doc_path = result.stdout.strip()
        if not self.doc_path:
            return
        self.save_config(property_name='documents_path', data=self.doc_path)
        
        yield self.doc_path, gr.update(value="Loading documents..."), gr.update(value=self.update_progress_bar(0))

        for update, percent in self.load_or_create_index(self.doc_path):
            yield self.doc_path, gr.update(value=update), gr.update(value=self.update_progress_bar(percent))

        yield self.doc_path, gr.update(value="FAISS Index successfully loaded!"), gr.update(value=self.update_progress_bar(100))
        
    def load_documents(self, folder_path):
        """
        Load documents from the specified folder path.

        :param folder_path: The path to the folder containing data.
        :return: A list of loaded documents.
        """
        try:
            if os.path.exists(folder_path) and os.listdir(folder_path):
                file_metadata = lambda x: {"filename": x}
                documents = SimpleDirectoryReader(folder_path, file_metadata=file_metadata,
                                                  recursive=True,
                                                  raise_on_error=True,
                                                  required_exts=[".pdf", ".doc", ".docx", ".txt", ".xml"]).load_data()
            else:
                documents = []
        except Exception as e:
            documents = []
        return documents
    
    def get_embedding_model(self):
        try:
            if self.config and "embedding_model" in self.config:
                model, dimension = self.config['embedding_model'].split(':')
                embedding_model = HuggingFaceEmbedding(model_name=model)
                
                return embedding_model, int(dimension)
        except Exception as e:
            message = f"Error loading embedding model: {e}"
            print(message, file=sys.stderr)
            gr.Error(message=message)
            return None, None
    
    def load_or_create_index(self, folder_path):
        persist_dir = f"{folder_path}_faiss_index"
        
        try:
            embedding_model, dimension = self.get_embedding_model()
            
            if not embedding_model or not dimension:
                raise "Unabled to load embedding model"
            
            if os.path.exists(persist_dir) and os.listdir(persist_dir):
                vector_store = FaissVectorStore.from_persist_dir(persist_dir)
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store, persist_dir=persist_dir
                )
                self.index = load_index_from_storage(storage_context=storage_context, embed_model=embedding_model)
            else:
                documents = self.load_documents(folder_path)
                file_list = set(doc.metadata.get("file_name", "Unknown") for doc in documents)
                num_files = len(file_list)
                progress = gr.Progress(track_tqdm=True)
                for i, file in enumerate(progress.tqdm(file_list, desc="Processing Files", total=num_files)):
                    percentage = int(i / num_files * 100)
                    yield f"Processing file {i + 1}/{num_files}: {file}", percentage
                    faiss_index = faiss.IndexFlatL2(dimension)
                    vector_store = FaissVectorStore(faiss_index=faiss_index)
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    self.index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embedding_model)
                    self.index.storage_context.persist(persist_dir=persist_dir)

                yield f"FAISS Index created with {num_files} files!", 100
        except Exception as e:
            message = f"Error creating FAISS index: {e}"
            print(message, file=sys.stderr)
            gr.Error(message=message)

    def chat_with_ollama(self, message):
        """Sends a message to Ollama and streams the response."""
        self.chat_initiated = True
        if not isinstance(self.chat_history, list):
            self.chat_history = []
            
        source_info = ""

        if self.ollama_mode == "Search":
            self.load_or_create_index(self.doc_path)  

            if self.index:
                query_engine = self.index.as_query_engine(streaming=False, similarity_top_k=self.top_doc)
                
                response = query_engine.query(message)
              
                context = ""
                
                if response:
                    context = str(response)
                    
                    files = []
                    if response.metadata:
                        for _, info in response.metadata.items():
                            files.append({'document': info.get("file_name", "Unknown File"), 'page': info.get("page_label", "Unknown Page") })
    
                        if len(files):
                            source_info += "\n\nReferenced files:"
                            for item in files:
                                entry = f"\n\t{item['document']}, p.{item['page']}"
                                if entry not in source_info:
                                    source_info += entry
                        
                if context == "":
                    context = "No documents have been loaded. Notify the user to try and reload the documents."
                
                ollama_prompt = f"""You are an AI assistant using retrieved documents to answer questions. 
                Here is the relevant information retrieved from the knowledge base:

                {context}

                Now, answer the following user query based on this information. You may also use your own knowledge to provide greater detail when appropriate.
                """
                
                self.chat_history.append({"role": "system", "content": ollama_prompt})
                
        self.chat_history.append({"role": "user", "content": message})
        
        if not self.summary or self.summary == "":
            self.set_summary(message)
        
        self.save_chat_history()
                
        response_text = ""
        try:
            if self.client and self.model:
                for chunk in self.client.chat(
                        model=self.model,
                        messages=self.chat_history,
                        stream=True
                    ):
                    if "message" in chunk:
                        response_text += chunk["message"]["content"]  # Accumulate response

                        if self.chat_history and self.chat_history[-1]["role"] == "assistant":
                            self.chat_history[-1]["content"] = response_text
                        else:
                            self.chat_history.append({"role": "assistant", "content": response_text})

                        yield self.chat_history, gr.update(value="")
            else:
                raise Exception("Ollama client not loaded")
        except ollama.RequestError as e:
            self.chat_history.append({"role": "assistant", "content": f"⚠️ Error: {str(e)}"})
            yield self.chat_history, gr.update(value="")
        except Exception as e:
            message = f"Error making request to Ollama: {e}"
            print(message, file=sys.stderr)
            gr.Error(message=message)
            sys.exit(1)
            
        if self.ollama_mode == "Search" and len(source_info):
            self.chat_history[-1]["content"] += source_info
            yield self.chat_history, gr.update(value="")
        
    def update_mode(self, value):
        self.ollama_mode = value
        isSearchMode = self.ollama_mode == "Search"
        return gr.update(value=self.ollama_mode), gr.update(visible=isSearchMode), gr.update(visible=isSearchMode)

    def detect_clear(self, chatbot):
        value = chatbot
        if not value or len(value) == 0:
            self.save_chat_history()
            self.session_timestamp = None
            self.current_save_file = None
            self.summary = ""
            self.chat_history = []

    def launch_ui(self):
        if self.config and "default_model" in self.config and self.config['default_model'] is not None and self.config['default_model'] != '':
            default_model = self.config['default_model']
            self.model = default_model
        elif self.models:
            default_model = self.models[0]
            self.model = default_model
        else:
            print("No models found.\nDownload models using the following command:\n\tollama pull <model_name>")
            sys.exit(1)

        with gr.Blocks() as app:
            gr.HTML(self.styles)
            gr.Markdown(f"## 🔥 {self.title} 🔥")
            chat_modes = ["Chat", "Search"]
            chat_state = gr.State(value=self.chat_history)
            chat_file = gr.State(value=self.current_save_file)

            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Row():
                        model_dropdown = gr.Dropdown(choices=self.models, label="Select Model", value=default_model, interactive=True)
                        model_dropdown.change(fn=self.update_default_model, inputs=[model_dropdown], outputs=[])
                        mode_dropdown = gr.Dropdown(choices=chat_modes, label="Select Mode", value=self.ollama_mode, interactive=True)
                        
                        folder_display = gr.Textbox(value=self.doc_path, label="Folder:", interactive=False, placeholder="Please select a folder", submit_btn="Select Folder", visible=(self.ollama_mode == "Search"))
                        progress_container = gr.Column(elem_classes=["progress-container"], visible=(self.ollama_mode == "Search"))
                        with progress_container:
                            progress_display = gr.Textbox(container=False, max_lines=1, text_align='left', interactive=False, elem_classes=["progress-display"])
                            progress_bar = gr.HTML(padding=False, value=self.update_progress_bar(None), elem_classes=["progress-bar"])
                        mode_dropdown.change(fn=self.update_mode, inputs=[mode_dropdown], outputs=[mode_dropdown, folder_display, progress_container])
                        folder_display.submit(fn=self.select_folder, inputs=[], outputs=[folder_display, progress_display, progress_bar], show_progress='hidden')
                        
                    chatbot = gr.Chatbot(type="messages", height=800, value=self.chat_history, layout=self.layout, sanitize_html=False, autoscroll=True)
                    chatbot.change(fn=self.detect_clear, inputs=[chatbot], outputs=[])
                    input_textbox = gr.Textbox(label="Chat with Ollama", interactive=True)
                    input_textbox.submit(self.chat_with_ollama, [input_textbox], [chatbot, input_textbox])
                    
                history_sidebar = gr.Sidebar(width=300, position='right')
                with history_sidebar:
                    gr.Markdown(value="### Chat History")
                    if self.chat_list:
                        for item in self.chat_list:
                            with gr.Row():
                                select_chat = gr.Button(value=item.summary)
                                select_chat.click(fn=self.update_chat_history, inputs=[gr.State(value=item.history), gr.State(value=item.filename)], outputs=[chatbot])
                                
            chatbot.change(fn=self.detect_clear, inputs=[chatbot], outputs=[])
                                
        app.launch(server_name=self.gradio_server_name, server_port=self.gradio_server_port, inbrowser=self.inbrowser, quiet=True, share=False)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    chatbot = OllamaChatbot()
    signal.signal(signal.SIGINT, chatbot.signal_handler)       
    chatbot.launch_ui()        

if __name__ == "__main__":
    main()
