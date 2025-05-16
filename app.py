from http.client import responses
from flask_cors import CORS
from flask import Flask, request, jsonify
from datetime import datetime
import json
import requests
import re
import os

MEMORY_FILE = "memory.json"

app = Flask(__name__)
CORS(app)

# In-memory chat history
chat_history = []

def update_memory_from_prompt(prompt):
    # Load existing memory
    memory = {}
    if os.path.exists("MEMORY_FILE"):
        with open("MEMORY_FILE", "r") as f:
            memory = json.load(f)
    updated = False

    # Patterns to match known user facts
    name_match = re.search(r"(ma nem is|call me)\s+([a-zA-Z]+)", prompt, re.IGNORECASE)
    location_match = re.search(r"(i am from|i live ina)\s+([a-zA-Z]+)", prompt, re.IGNORECASE)
    likes_match = re.search(r"i like\s+([a-zA-Z]+)", prompt, re.IGNORECASE)

    if name_match:
        memory["user_name"] = name_match.group(2).strip()
        updated = True
    if location_match:
        memory["location"] = location_match.group(2).strip()
        updated = True
    if likes_match:
        memory.setdefault("likes", [])
        new_likes = [item.strip() for item in likes_match.group(1).split(",")]
        memory["likes"] = list(set(memory["likes"] + new_likes))
        updated = True

    if updated:
        with open("MEMORY_FILE", "w") as f:
            json.dump(memory, f, indent=2)
        print("[MEMORY] Updated memory:", memory)

def ask_assistant(prompt, task="chat"):
    model_map ={
        "chat":"llama3",
        "summarize":"mistral",
        "code":"starcoder",
        "casual":"mistral"
    }
    model = model_map.get(task, "llama3")

    # Load memory
    memory_text = ""
    try:
        with open("memory.json", "r") as f:
            memory = json.load(f)
            memory_text += f"The user's name is {memory.get('user_name', 'unknown')}.\n"
            memory_text += f"The assistant's name is {memory.get('name', 'Assistant')}\n"
            goals = memory.get("goals", [])
            if goals:
                memory_text += f"The assistant's goal's are :\n" +"\n".join(f" - {g}" for g in goals)+"\n"
    except Exception as e:
        print("Could not load memory")

    # Construct prompt with memory context
    full_prompt = f"{memory_text}\nUser: {prompt}\nAssistant:"

    if model == "llama3":
        formatted_prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
    elif task in ["chat", "casual"]:
        formatted_prompt = f"You are a helpful assistant.\nUser:{prompt}\nAssistant:" if task =="chat" else prompt
    else:
        formatted_prompt = full_prompt

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

    update_memory_from_prompt(prompt)

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
@app.route('/memory', methods=['POST'])
def update_memory():
    data = request.get_json()
    with open("memory.json", "w") as f:
        json.dump(data, f, indent=2)
    return jsonify({"status":"memory updated"})

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)