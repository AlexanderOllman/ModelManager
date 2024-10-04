from flask import Flask, render_template, jsonify, request
import subprocess
import json
import yaml
import threading
import time
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

deployment_lock = threading.Lock()
deployment_in_progress = False

def run_kubectl_command(command):
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    return result.stdout, result.stderr

def check_runtime_exists(runtime_name):
    _, error = run_kubectl_command(f"kubectl get clusterservingruntime {runtime_name}")
    return "NotFound" not in error

def deploy_runtime(runtime_yaml, runtime_name):
    global deployment_in_progress
    
    with open('runtime.yaml', 'w') as f:
        yaml.dump(runtime_yaml, f)
    
    result, error = run_kubectl_command("kubectl apply -f runtime.yaml")
    if error:
        socketio.emit('deployment_status', {'status': 'error', 'message': f"Failed to deploy runtime: {error}"})
        deployment_in_progress = False
        return False
    
    socketio.emit('deployment_status', {'status': 'info', 'message': "Runtime deployed. Waiting for 15 seconds..."})
    time.sleep(15)
    
    for i in range(3):
        if check_runtime_exists(runtime_name):
            socketio.emit('deployment_status', {'status': 'success', 'message': "Runtime is ready."})
            return True
        if i < 2:
            socketio.emit('deployment_status', {'status': 'info', 'message': f"Runtime not ready. Checking again in 10 seconds... (Attempt {i+1}/3)"})
            time.sleep(10)
    
    socketio.emit('deployment_status', {'status': 'error', 'message': "Runtime deployment failed after 3 attempts."})
    deployment_in_progress = False
    return False

def deploy_inference_service(inference_yaml, namespace):
    with open('inference.yaml', 'w') as f:
        yaml.dump(inference_yaml, f)
    
    result, error = run_kubectl_command(f"kubectl apply -f inference.yaml -n {namespace}")
    if error:
        socketio.emit('deployment_status', {'status': 'error', 'message': f"Failed to deploy inference service: {error}"})
        return False
    
    socketio.emit('deployment_status', {'status': 'success', 'message': "Inference service deployed successfully."})
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models')
def models():
    isvc_data, _ = run_kubectl_command("kubectl get isvc -A -o json")
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
    isvc_data, _ = run_kubectl_command("kubectl get isvc -A -o json")
    return jsonify(json.loads(isvc_data))

@app.route('/api/clusterservingruntime')
def get_clusterservingruntime():
    csr_data, _ = run_kubectl_command("kubectl get clusterservingruntime -A -o json")
    return jsonify(json.loads(csr_data))

@app.route('/api/deploy', methods=['POST'])
def deploy_model():
    global deployment_in_progress
    
    if deployment_in_progress:
        return jsonify({'status': 'error', 'message': 'A deployment is already in progress. Please wait.'})
    
    with deployment_lock:
        if deployment_in_progress:
            return jsonify({'status': 'error', 'message': 'A deployment is already in progress. Please wait.'})
        deployment_in_progress = True
    
    data = request.json
    inference_yaml = yaml.safe_load(data['inferenceYaml'])
    runtime_yaml = yaml.safe_load(data['runtimeYaml'])
    runtime_name = data['runtime']
    namespace = data['namespace']
    
    socketio.emit('deployment_status', {'status': 'info', 'message': "Starting deployment process..."})
    
    if not check_runtime_exists(runtime_name):
        socketio.emit('deployment_status', {'status': 'info', 'message': "Runtime doesn't exist. Deploying runtime..."})
        if not deploy_runtime(runtime_yaml, runtime_name):
            deployment_in_progress = False
            return jsonify({'status': 'error', 'message': 'Runtime deployment failed.'})
    else:
        socketio.emit('deployment_status', {'status': 'info', 'message': "Runtime already exists. Proceeding with inference service deployment..."})
    
    if deploy_inference_service(inference_yaml, namespace):
        deployment_in_progress = False
        return jsonify({'status': 'success', 'message': 'Model deployed successfully.'})
    else:
        deployment_in_progress = False
        return jsonify({'status': 'error', 'message': 'Inference service deployment failed.'})

@app.route('/api/delete-model', methods=['POST'])
def delete_model():
    data = request.json
    model_name = data['modelName']
    namespace = data['namespace']

    result, error = run_kubectl_command(f"kubectl delete isvc {model_name} -n {namespace}")

    if error:
        return jsonify({
            'success': False,
            'error': f"Failed to delete model: {error}"
        })

    return jsonify({
        'success': True,
        'result': result
    })

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port='8080')