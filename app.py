from flask import Flask, render_template, jsonify, request
import subprocess
import json
import requests

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

@app.route('/deploy')
def deploy():
    return render_template('deploy.html')

@app.route('/api/isvc')
def get_isvc():
    isvc_data = run_kubectl_command("kubectl get isvc -A -o json")
    return jsonify(json.loads(isvc_data))

@app.route('/api/clusterservingruntime')
def get_clusterservingruntime():
    csr_data = run_kubectl_command("kubectl get clusterservingruntime -A -o json")
    return jsonify(json.loads(csr_data))

@app.route('/api/ai-applications')
def get_ai_applications():
    domain = request.args.get('domain')
    if not domain:
        return jsonify({"error": "Domain parameter is required"}), 400
    
    url = f"https://home.{domain}/api/v1/ai-applications"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8080')