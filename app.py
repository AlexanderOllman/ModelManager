from flask import Flask, render_template, jsonify, request, Response
import subprocess
import json
import yaml
import threading
import time
import os
import logging
from flask_socketio import SocketIO
from huggingface_hub import snapshot_download

app = Flask(__name__)
socketio = SocketIO(app)

deployment_lock = threading.Lock()
deployment_in_progress = False

logging.basicConfig(level=logging.ERROR)

def run_kubectl_command(command):
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    return result.stdout, result.stderr

def check_runtime_exists(runtime_name):
    _, error = run_kubectl_command(f"kubectl get clusterservingruntime {runtime_name}")
    return "NotFound" not in error

def check_deployment_status(namespace, model_name):
    max_attempts = 30
    attempt = 0
    while attempt < max_attempts:
        # Check pod status
        pod_command = f"kubectl get pods -n {namespace} -l serving.kserve.io/inferenceservice={model_name} -o json"
        pod_output, pod_error = run_kubectl_command(pod_command)
        if pod_error:
            socketio.emit('deployment_status', {'status': 'error', 'message': f"Error checking pod status: {pod_error}"})
            return

        pod_data = json.loads(pod_output)
        if pod_data['items']:
            pod_status = pod_data['items'][0]['status']['phase']
            socketio.emit('deployment_status', {'status': 'info', 'message': f"Pod status: {pod_status}"})
            
            if pod_status == 'Running':
                # Check InferenceService status
                isvc_command = f"kubectl get inferenceservice {model_name} -n {namespace} -o json"
                isvc_output, isvc_error = run_kubectl_command(isvc_command)
                if isvc_error:
                    socketio.emit('deployment_status', {'status': 'error', 'message': f"Error checking InferenceService status: {isvc_error}"})
                    return

                isvc_data = json.loads(isvc_output)
                conditions = isvc_data.get('status', {}).get('conditions', [])
                ready_condition = next((c for c in conditions if c['type'] == 'Ready'), None)
                
                if ready_condition:
                    socketio.emit('deployment_status', {'status': 'info', 'message': f"InferenceService status: {ready_condition['status']}"})
                    
                    if ready_condition['status'] == 'True':
                        socketio.emit('deployment_status', {'status': 'success', 'message': "Model deployed successfully!", 'final': True})
                        return

        attempt += 1
        time.sleep(10)  # Wait for 10 seconds before checking again

    socketio.emit('deployment_status', {'status': 'error', 'message': "Deployment timed out", 'final': True})

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
    
    socketio.emit('deployment_status', {'status': 'info', 'message': "Inference service deployment started."})
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

@app.route('/download')
def download():
    return render_template('download.html')

@app.route('/api/isvc')
def get_isvc():
    isvc_data, _ = run_kubectl_command("kubectl get isvc -A -o json")
    return jsonify(json.loads(isvc_data))

@app.route('/api/clusterservingruntime')
def get_clusterservingruntime():
    csr_data, _ = run_kubectl_command("kubectl get clusterservingruntime -A -o json")
    return jsonify(json.loads(csr_data))

@app.route('/api/pods')
def get_pods():
    command = "kubectl get pods -A -l serving.kserve.io/inferenceservice -o json"
    pods_data, error = run_kubectl_command(command)
    if error:
        return jsonify({'error': str(error)}), 500
    return pods_data

@app.route('/api/pod-logs/<namespace>/<pod_name>')
def get_pod_logs(namespace, pod_name):
    command = f"kubectl logs -n {namespace} {pod_name}"
    logs, error = run_kubectl_command(command)
    if error:
        return str(error), 500
    return logs

@app.route('/api/namespaces')
def get_namespaces():
    namespaces_data, _ = run_kubectl_command("kubectl get namespaces -o json")
    namespaces_json = json.loads(namespaces_data)
    return jsonify([item['metadata']['name'] for item in namespaces_json['items']])

@app.route('/api/gpu-info')
def get_gpu_info():
    gpu_info_command = """
    kubectl get nodes -o custom-columns='NODE:.metadata.name,GPU_COUNT:.status.allocatable.nvidia\.com/gpu,GPU_MODEL:.metadata.labels.nvidia\.com/gpu\.product,GPU_MEMORY:.metadata.labels.nvidia\.com/gpu\.memory' | grep -v '<none>' | grep -v ' 0 ' | sed -E 's/^([^ ]+) +([^ ]+) +([^ ]+) +([^ ]+)$/{"nodeName": "\\1", "gpuCount": "\\2", "gpuModel": "\\3", "gpuMemory": "\\4"}/'
    """
    gpu_info_data, error = run_kubectl_command(gpu_info_command)
    if error:
        logging.error(f"Error fetching GPU info: {error}")
        return jsonify([])
    
    try:
        gpu_info_list = [json.loads(line) for line in gpu_info_data.strip().split('\n') if line.strip()]
        logging.error(f"GPU Info: {gpu_info_list}")  # Log the result for debugging
        return jsonify(gpu_info_list)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing GPU info: {e}")
        logging.error(f"Raw GPU info data: {gpu_info_data}")
        return jsonify([])

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
    model_name = data['modelName']
    
    socketio.emit('deployment_status', {'status': 'info', 'message': "Starting deployment process..."})
    
    if not check_runtime_exists(runtime_name):
        socketio.emit('deployment_status', {'status': 'info', 'message': "Runtime doesn't exist. Deploying runtime..."})
        if not deploy_runtime(runtime_yaml, runtime_name):
            deployment_in_progress = False
            return jsonify({'status': 'error', 'message': 'Runtime deployment failed.'})
    else:
        socketio.emit('deployment_status', {'status': 'info', 'message': "Runtime already exists. Proceeding with inference service deployment..."})
    
    if deploy_inference_service(inference_yaml, namespace):
        threading.Thread(target=check_deployment_status, args=(namespace, model_name)).start()
        return jsonify({'status': 'info', 'message': 'Deployment started. Checking status...'})
    else:
        deployment_in_progress = False
        return jsonify({'status': 'error', 'message': 'Inference service deployment failed.'})

@app.route('/api/delete/<resource_type>/<resource_name>', methods=['DELETE'])
def delete_resource(resource_type, resource_name):
    namespace = request.args.get('namespace')
    if resource_type == 'isvc':
        command = f"kubectl delete inferenceservice {resource_name}"
        if namespace:
            command += f" -n {namespace}"
    elif resource_type == 'csr':
        command = f"kubectl delete clusterservingruntime {resource_name}"
    else:
        return jsonify({'success': False, 'error': 'Invalid resource type'}), 400

    output, error = run_kubectl_command(command)
    if error:
        return jsonify({'success': False, 'error': error}), 500
    return jsonify({'success': True, 'message': f'{resource_type.upper()} {resource_name} deleted successfully'})

@app.route('/api/describe/<resource_type>/<resource_name>')
def describe_resource(resource_type, resource_name):
    namespace = request.args.get('namespace')
    if resource_type == 'isvc':
        command = f"kubectl describe inferenceservice {resource_name}"
        if namespace:
            command += f" -n {namespace}"
    elif resource_type == 'csr':
        command = f"kubectl describe clusterservingruntime {resource_name}"
    else:
        return jsonify({'error': 'Invalid resource type'}), 400

    description, error = run_kubectl_command(command)
    if error:
        return jsonify({'error': error}), 500
    return jsonify({'description': description})

@app.route('/api/list-models')
def list_models():
    models_path = "/mnt/models/hub"
    try:
        if not os.path.exists(models_path):
            return jsonify([])
        
        model_paths = []
        for root, dirs, _ in os.walk(models_path):
            if root != models_path:  # Skip the base directory
                rel_path = os.path.relpath(root, models_path)
                if rel_path != ".":  # Skip current directory
                    model_paths.append(rel_path)
        
        return jsonify(model_paths)
    except Exception as e:
        print(f"Error listing models: {e}")
        return jsonify([])

@app.route('/api/download-model')
def download_model():
    def generate():
        try:
            model_repo = request.args.get('repo')
            if not model_repo:
                yield "data: Error: No model repository specified\n\n"
                return

            cache_dir = "/mnt/models/hub"
            yield f"data: Starting download of {model_repo}\n\n"
            
            snapshot_download(
                repo_id=model_repo,
                cache_dir=cache_dir,
                local_files_only=False,
                allow_patterns=["*.safetensors", "*.json"]
            )
            
            yield f"data: Model files downloaded to {cache_dir}\n\n"
        except Exception as e:
            yield f"data: Error downloading model: {str(e)}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port='8080')