from flask import Flask, render_template, jsonify, request
import subprocess
import json
import yaml

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

@app.route('/api/deploy', methods=['POST'])
def deploy_model():
    data = request.json
    inference_yaml = yaml.safe_load(data['inferenceYaml'])
    runtime_yaml = yaml.safe_load(data['runtimeYaml'])

    # Save YAML files
    with open('inference.yaml', 'w') as f:
        yaml.dump(inference_yaml, f)
    with open('runtime.yaml', 'w') as f:
        yaml.dump(runtime_yaml, f)

    # Apply YAML files using kubectl
    inference_result = run_kubectl_command("kubectl apply -f inference.yaml")
    runtime_result = run_kubectl_command("kubectl apply -f runtime.yaml")

    return jsonify({
        'inference_result': inference_result,
        'runtime_result': runtime_result
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8080')