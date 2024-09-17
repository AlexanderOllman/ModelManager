from flask import Flask, render_template, jsonify
import subprocess
import json

app = Flask(__name__)

def run_kubectl_command(command):
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    return result.stdout

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models')
def models():
    isvc_data = run_kubectl_command("kubectl get isvc -A -o json")
    isvc_json = json.loads(isvc_data)
    return render_template('models.html', isvc_data=isvc_json['items'])

@app.route('/manage')
def manage():
    return render_template('manage.html')

@app.route('/api/isvc')
def get_isvc():
    isvc_data = run_kubectl_command("kubectl get isvc -A -o json")
    return jsonify(json.loads(isvc_data))

@app.route('/api/clusterservingruntime')
def get_clusterservingruntime():
    csr_data = run_kubectl_command("kubectl get clusterservingruntime -A -o json")
    return jsonify(json.loads(csr_data))

if __name__ == '__main__':
    app.run(debug=True)