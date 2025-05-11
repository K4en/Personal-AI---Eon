from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)
def ask_assistant(prompt):
    result = subprocess.run(
        ['ollama', 'run', 'llama2', '--prompt',prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.stdout.script()

@app.route("/ask", methods=["POST"])
def ask():
    data= request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error:""Missing prompt"}),400
    response = ask_assistant(prompt)
    return jsonify({"response":response})

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)