from http.client import responses
from flask_cors import CORS
from flask import Flask, request, jsonify
from datetime import datetime
import json
import requests
import re
import os
import subprocess

MEMORY_FILE = "memory.json"
KNOWLEDGE_FILE = "knowledge.jsonl"
TASK_FILE = "tasks.json"
HISTORY_FILE = "chat_history.jsonl"
PENDING_TASK_FILE = "pending_task.json"
GOALS_FILE = "goals.json"
REMINDERS_FILE = "reminders.jsonl"
JOURNAL_FILE = "journal.jsonl"
IDENTITY_FILE = "identity.json"
CONTEXT_FILE = "context.json"
ACTIVE_THREAD_FILE = "active_thread.json"
app = Flask(__name__)
CORS(app)

# In-memory chat history
chat_history = []
document_chunks = []

if os.path.exists(KNOWLEDGE_FILE):
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        document_chunks = [json.loads(line.strip()) for line in f if line.strip()]

def load_goals():
    if os.path.exists(GOALS_FILE):
        with open(GOALS_FILE, "r") as f:
            return json.load(f)
    return []

def load_pending_task():
    if os.path.exists(PENDING_TASK_FILE):
        with open(PENDING_TASK_FILE, "r") as f:
            return json.load(f)
    return None

def save_pending_task(data):
    with open(PENDING_TASK_FILE, "w") as f:
        json.dump(data, f, indent=2)

def clear_pending_task():
    if os.path.exists(PENDING_TASK_FILE):
        os.remove(PENDING_TASK_FILE)

def load_tasks():
    if os.path.exists(TASK_FILE):
        with open(TASK_FILE, "r") as f:
            return json.load(f)
    return []

def run_task_internal(task_name, user_args=[]):
    tasks = load_tasks()
    task = next((t for t in tasks if t["name"] == task_name), None)
    if not task:
        return jsonify({"error": "Task not found"}), 404

    command = task["command"]
    # Handle internal commands
    if command == "list_dir":
        if not user_args:
            return jsonify({"error": "Missing folder path"}), 400
        folder = user_args[0]
        try:
            files = os.listdir(folder)
            return jsonify({"files": files})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    elif command == "read_file":
        if not user_args:
            return jsonify({"error": "Missing file path"}), 400
        filepath = user_args[0]
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            return jsonify({"content": content})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # External OS-level commands
    full_command = [command] + task.get("args", []) + user_args
    try:
        subprocess.Popen(full_command, shell=True)
        return jsonify({"status":"task started","command": full_command})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def summarize_for_memory(prompt, response):
    summary_prompt = (
        "Extract important facts from this conversation that the assistant should remember about the user."
        "Respond with a JSON object or say 'None' if there's nothing to remember.\n\n"
        f"User:{prompt}\nAssistant:{response}"
    )
    summary_payload = {
        "model": "llama3",
        "prompt": summary_prompt,
        "stream": False
    }
    # Ollama API Url
    url = "http://localhost:11434/api/generate"

    try:
        result = requests.post(url, json=summary_payload)
        result.raise_for_status()
        raw = result.json().get("response", "").strip()
        print("[SUMMARY RAW RESPONSE]", raw)

        if "None" not in raw:
            summary_data = json.loads(raw)
            return summary_data
    except Exception as e:
        print(f"[SUMMARY ERROR] {e}")
    return None

def update_memory_from_prompt(prompt):
    # Load existing memory
    memory = {}
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            memory = json.load(f)
    updated = False

    # Patterns to match known user facts
    name_match = re.search(r"(my name is|call me)\s+([a-zA-Z]+)", prompt, re.IGNORECASE)
    location_match = re.search(r"(i am from|i live in)\s+([a-zA-Z]+)", prompt, re.IGNORECASE)
    likes_match = re.search(r"i like\s+([a-zA-Z]+)", prompt, re.IGNORECASE)

    if name_match:
        memory["user_name"] = {
            "value": name_match.group(2).strip(),
            "type": "identity"
        }
        updated = True
    if location_match:
        memory["location"] = {
            "value": location_match.group(2).strip(),
            "type": "preference"
        }
        updated = True
    if likes_match:
        memory.setdefault("likes", [])
        new_likes = [item.strip() for item in likes_match.group(1).split(",")]
        memory["likes"] = {
            "value": list(set(memory["likes"] + new_likes)),
            "type": "interest"
        }
        updated = True

    if updated:
        with open(MEMORY_FILE, "w") as f:
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
    memory = {}
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            memory = json.load(f)

    memory_intro = ""
    for key, data in memory.items():
        memory_intro += f"{key.replace('_', ' ').capitalize()}: {data['value']}\n"

    # Recent thread history
    thread_name = "general"
    if os.path.exists(ACTIVE_THREAD_FILE):
        with open(ACTIVE_THREAD_FILE, "r") as f:
            thread_name = json.load(f).get("name", "general")

    thread_path = f"threads/{thread_name}.jsonl"
    thread_history = []
    if os.path.exists(thread_path):
        with open(thread_path, "r", encoding="utf-8") as f:
            thread_history = [json.loads(line) for line in f]

    context = ""
    for entry in thread_history:
        context += f"User: {entry['prompt']}\nAssistant: {entry['response']}\n"

    # Load current context mode
    context_mode = "default"
    if os.path.exists(CONTEXT_FILE):
        with open(CONTEXT_FILE, "r") as f:
            context_mode = json.load(f).get("mode", "default")
    context_intro = f"[Current Mode: {context_mode}]\n"

    # Construct prompt with memory context
    full_prompt = f"{context_intro}{memory_intro}\nRecent conversation:\n{context}User: {prompt}\nAssistant:"

    if model == "llama3":
        formatted_prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{full_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
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
        ai_reply = response.json().get("response", "").strip()

        #Save to current thread
        with open(thread_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": ai_reply
            })+"\n")
        return ai_reply
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

    # If a pending task waiting, treat promp as argument
    pending = load_pending_task()
    if pending:
        # Inject the user's clarification into the args
        pending["args"] = [prompt]
        clear_pending_task()
        return run_task_internal(pending["task"], pending["args"])

    update_memory_from_prompt(prompt)

    response = ask_assistant(prompt, task)

    summary = summarize_for_memory(prompt, response)



    if summary:
        with open(MEMORY_FILE, "r+") as f:
            memory = json.load(f)
            for k, v in summary.items():
                if isinstance(v, dict) and "value" in v:
                    memory[k] = v
                else:
                    memory[k] = {"value": v, "type": "auto"}
            f.seek(0)
            json.dump(memory, f, indent=2)
            f.truncate()
            print("[MEMORY] Auto-updated with:", summary)

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
    with open(HISTORY_FILE, "a",encoding="utf-8") as f:
       f.write(json.dumps(entry)+"\n")

    return jsonify({"response":response})

@app.route('/history', methods=['GET'])
def history():
    return jsonify(chat_history)
@app.route('/memory', methods=['POST'])
def update_memory():
    data = request.get_json()
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return jsonify({"status":"memory updated"})

@app.route('/memory', methods=['GET'])
def get_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            memory = json.load(f)
        return jsonify(memory)
    return jsonify({})

@app.route('/add_knowledge', methods=['POST'])
def add_knowledge():
    data = request.get_json()
    content = data.get("content", "")
    tags = data.get("tags", [])
    if not content:
        return jsonify({"error": "No content provided"}), 400

    entry = {"content":content, "tags": tags}
    document_chunks.append(entry)

    # Save to file
    with open (KNOWLEDGE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry)+"\n")

    return jsonify({"status": "added", "chunks": len(document_chunks)})

@app.route('/knowledge', methods=['GET'])
def get_knowledge():
    if not document_chunks:
        return jsonify([])

    return jsonify(document_chunks)

@app.route('/knowledge/<int:index>', methods=['PUT'])
def update_knowledge(index):
    data = request.get_json()
    content = data.get("content", "")
    tags = data.get("tags", [])

    if 0 <= index < len(document_chunks) and content:
        document_chunks[index] = {"content":content, "tags": tags}
        with open (KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
            for chunk in document_chunks:
                f.write(chunk+"\n")
        return jsonify({"status": "updated", "chunks": len(document_chunks)})
    return jsonify({"error": "Invalid index or content"}), 400

@app.route('/knowledge/<int:index>', methods=['DELETE'])
def delete_knowledge(index):
    if 0 <= index < len(document_chunks):
        del document_chunks[index]
        with open (KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
            for chunk in document_chunks:
                f.write(chunk+"\n")
        return jsonify({"status": "deleted", "chunks": len(document_chunks)})
    return jsonify({"error": "Invalid index"}), 400

@app.route('/run_task', methods=['POST'])
def run_task():
    data = request.get_json()
    task_name = data.get("task", "")
    user_args = data.get("args", [])
    confirm = data.get("confirm", False)

    tasks = load_tasks()
    task = next((t for t in tasks if t["name"] == task_name), None)

    if not task:
        return jsonify({"error": "Task not found"}), 404

    if task.get("requires_confirmation", False) and not confirm:
        return jsonify({"error": "Task requires confirmation"}), 400

    if task.get("requires_confirmation", False) and not user_args:
        save_pending_task({"task": task_name})
        return jsonify({
            "clarification_needed": True,
            "message":f"What should I {task_name.replace('_', ' ').title()}?"
        }), 400

    # Combine task args + user args if required
    command = [task["command"]] + task.get("args", [])
    if task.get("requires_args"):
        if not user_args:
            return jsonify({"error": "This task requires additional input"}), 400
        command += user_args


    try:
        subprocess.Popen(command, shell=True)
        return jsonify({"status": "task started", "command":command})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Learn task rout
@app.route('/learn_task', methods=['POST'])
def learn_task():
    data = request.get_json()
    name = data.get("name", "").strip().lower().replace(" ", "_")
    command = data.get("command", "").strip()

    if not name or not command:
        return jsonify({"error": "Missing name or command"}), 400

    new_task = {
        "name": name,
        "description": "User defined short-cut",
        "command": command,
        "args": [],
        "requires_args": False
    }

    # Load existing tasks
    tasks = []
    if os.path.exists(TASK_FILE):
        with open(TASK_FILE, "r") as f:
            tasks = json.load(f)

    # Prevent duplicate
    if any(t["name"]== new_task["name"] for t in tasks):
        return jsonify({"error": "Task already exists"}), 409

    tasks.append(new_task)

    with open(TASK_FILE, "w") as f:
        json.dump(tasks, f, indent=2)

    return jsonify({"status": "task learned", "task": new_task})

@app.route('/run_goal', methods=['POST'])
def run_goal():
    data = request.get_json()
    goal_name = data.get("goal", "")

    goals = load_goals()
    goal = next((g for g in goals if g["goal"] == goal_name), None)

    if not goal:
        return jsonify({"error": "Goal not found"}), 404

    steps = goal.get("steps", [])
    for step in steps:
        result = run_task_internal(step["task"])
        if result.status_code != 200:
            return jsonify({"error": f"Failed step: {step['task']}"})

    return jsonify({"status": f"Goal '{goal_name}' completed", "steps": steps})

@app.route('/inject_knowledge', methods=['POST'])
def inject_knowledge():
    data = request.get_json()
    filepath = data.get("filepath", "").strip()
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "Invalid or missing filepath"}), 404

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps({"source": filepath, "content": content})+"\n")

    return jsonify({"status": "injected", "length": len(content)})

@app.route('/add_reminder', methods=['POST'])
def add_reminder():
    data = request.get_json()
    task = data.get("task", "").strip()
    time = data.get("time", "").strip()

    if not task or not time:
        return jsonify({"error": "Missing task or time"}), 400

    entry = {"task": task, "time": time, "status": "active"}

    with open(TASK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry)+"\n")

    return jsonify({"status": "reminder added", "reminder": entry})

@app.route('/reminders', methods=['GET'])
def list_reminders():
    reminders = []
    if os.path.exists(REMINDERS_FILE):
        with open(REMINDERS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    reminders.append(json.loads(line.strip()))
                except:
                    continue
    return jsonify({reminders})

@app.route('/add_journal', methods=['POST'])
def add_journal():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Missing text"}), 400

    entry = {
        "timestamp": datetime.now().isoformat(),
        "text": text
    }

    with open(JOURNAL_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry)+"\n")

    return jsonify({"status": "journal entry added", "entry": entry})

@app.route('/reflect', methods=['GET'])
def reflect():
    if not os.path.exists(JOURNAL_FILE):
        return jsonify({"error": "No journal entries"}), 404

    with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
        entries = [json.load(line.strip()) for line in f if line.strip()]

    full_text = "\n".join(f"[{e['timestamp']}]{e['text']}" for e in entries[-10:])

    prompt = (
        "Analyze the user's recent journal entries. Identify key emotional patterns, priorities or self-beliefs"
        "Respond with a reflection like a psychologist would - honest, insightful, nut not judgemental\n\n"
        f"{full_text}\n\nReflection:"
    )
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        summary = response.json().get("response", "").strip()
        return jsonify({"reflection": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/life_timeline', methods=['GET'])
def life_timeline():
    memory = {}
    journal_entries = []
    knowledge_entries = []

    # Load memory
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            memory = json.load(f)

    # Load journal
    if os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
            journal_entries = [json.load(line.strip()) for line in f if line.strip()]

    # Load knowledge
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            knowledge_entries = [json.loads(line.strip()) for line in f if line.strip()]

    combined_context =(
        "Memory:\n" + json.dumps(memory, indent=2) + "\n\n"+
        "Journal entries:\n" + "\n".join(f"[{e['timestamp']}] {e['text']}"for e in journal_entries[-10:]) + "\n\n"+
        "Knowledge entries:\n" + "\n".join(k.get("content","") for k in knowledge_entries[-5:])
    )

    prompt = (
        "Using the information provided, generate a structured timeline summarizing the user's life, key events, turning points, values, and mental evolution.\n\n"
        f"{combined_context}\n\nTimeline:"
    )
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        result = response.json().get("response", "").strip()
        return jsonify({"timeline": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/identify_self', methods=['GET'])
def identify_self():
    memory, journal = {}, []
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            memory = json.load(f)

    if os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
            journal = [json.loads(line.strip()) for line in f if line.strip()]

    sample = "\n".join(f"[{j['timestamp']}]{j['text']}]" for j in journal[-10:])
    prompt = (
        "Based on the user's memory and journal entries, generate a personality profile including tone, values, and conversational style."
        f"\n\nMemory: {json.dumps(memory, indent=2)}\n\nJournal:\n{sample}\n\nProfile:"
    )

    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        profile = response.json().get("response", "").strip()

        with open(IDENTITY_FILE, "w") as f:
            f.write(profile)

        return jsonify({"status": "profile generated", "profile": profile})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/context', methods=['POST'])
def set_context():
    data = request.get_json()
    mode = data.get("mode", "default")
    context = {"mode": mode, "last_switch": datetime.now().isoformat()}

    with open(CONTEXT_FILE, "w") as f:
        json.dump(context, f, indent=2)

    return jsonify({"status": f"context set to'{mode}'"})

@app.route('/context', methods=['GET'])
def get_context():
    if os.path.exists(CONTEXT_FILE):
        return jsonify({"mode": "default"})
    with open(CONTEXT_FILE, "r") as f:
        return jsonify(json.load(f))

@app.route('/thread', methods=['POST'])
def switch_thread():
    data = request.get_json()
    thread = data.get("name", "general")
    with open(ACTIVE_THREAD_FILE, "w") as f:
        json.dump({"name": thread,"last_switch": datetime.now().isoformat()}, f)
    return jsonify({"status": f"This '{thread}' is now active"})

@app.route('/thread', methods=['GET'])
def get_thread():
    if not os.path.exists(ACTIVE_THREAD_FILE):
        return jsonify({"name": "general"})
    with open(ACTIVE_THREAD_FILE, "r") as f:
        return jsonify(json.load(f))

@app.route('/goal_review', methods=['GET'])
def goal_review():
    if not os.path.exists(GOALS_FILE):
        return jsonify({"error": "No goals found"}), 404
    with open(GOALS_FILE, "r") as f:
        goals = json.load(f)

    #Load chat history (optional from thread or logs)
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = [json.loads(line.strip()) for line in f if line.strip()]

    last_20 = "\n".join(f"User: {e['prompt']}\nAI: {e['response']}" for e in history[-20:])

    prompt = (
        "Here is a list of long-term goals and recent conversation logs.\n"
        "Review which goals have progressed, which are stalled, and give suggestions.\n\n"
        f"Goals:\n{json.dumps(goals, indent=2)}\n\nRecent Conversation:\n{last_20}\n\nGoal Review:"
    )

    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }

    try:
        result = requests.post("http://localhost:11434/api/generate", json=payload)
        result.raise_for_status()
        return jsonify({"review": result.json().get("response", "").strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)