from http.client import responses
from flask_cors import CORS
from flask import Flask, request, jsonify
import subprocess
from datetime import datetime
import json
import requests

app = Flask(__name__)
CORS(app)

# In-memory chat history
chat_history = []

def ask_assistant(prompt, task="chat"):
    model_map ={
        "chat":"llama3",
        "summarize":"mistral",
        "code":"starcoder",
        "casual":"mistral"
    }
    model = model_map.get(task, "llama3")
    if model == "llama3":
        formatted_prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
    elif task in ["chat", "casual"]:
        formatted_prompt = f"You are a helpful assistant.\nUser:{prompt}\nAssistant:" if task =="chat" else prompt
    else:
        formatted_prompt = prompt

    print(f"[MODEL:{model}]:PROMPT:{formatted_prompt}")

    # Ollama API Url
    url = "http://localhost:11434/api/generate"

    # Post request to Ollama API
    payload={
        "model":model,
        "prompt":formatted_prompt,
        "stream":False
    }
    try:
        response=requests.post(url,json=payload)
        response.raise_for_status()
        return response.json().get("response","No response from model")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API request failed:{e}")
        return "Error contacting the model API"

@app.route("/ask", methods=["POST"])
def ask():
    data= request.get_json()
    prompt = data.get("prompt", "")
    task = data.get("task", "chat") # def to chat
    if not prompt:
        return jsonify({"error":"Missing prompt"}),400
    response = ask_assistant(prompt, task)

    # Log chat
    entry={
        "timestamp":datetime.now().isoformat(),
        "task":task,
        "prompt":prompt,
        "response":response
    }

    # Saving to memory
    chat_history.append(entry)

    # Append to file
    with open("chat_history.jsonl", "a",encoding="utf-8") as f:
       f.write(json.dumps(entry)+"\n")

    return jsonify({"response":response})

@app.route('/history', methods=['GET'])
def history():
    return jsonify(chat_history)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)