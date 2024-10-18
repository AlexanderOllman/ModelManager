from flask import Flask, render_template, jsonify, request
import subprocess
import json
import yaml
import threading
import time
from flask_socketio import SocketIO
import logging
import os

logging.basicConfig(level=logging.WARNING)

app = Flask(__name__)
socketio = SocketIO(app)

download_progress = {}

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

@app.route('/api/isvc')
def get_isvc():
    isvc_data, _ = run_kubectl_command("kubectl get isvc -A -o json")
    return jsonify(json.loads(isvc_data))

@app.route('/api/clusterservingruntime')
def get_clusterservingruntime():
    csr_data, _ = run_kubectl_command("kubectl get clusterservingruntime -A -o json")
    return jsonify(json.loads(csr_data))

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
        return jsonify({'error': error}), 400
    return jsonify({'description': description})

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


def get_dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

@app.route('/download')
def download_page():
    return render_template('download.html')

@app.route('/api/list-models')
def list_models():
    models_path = "/mnt/models"
    models = []
    try:
        for root, dirs, files in os.walk(models_path):
            for dir in dirs:
                relative_path = os.path.relpath(os.path.join(root, dir), models_path)
                models.append({
                    'name': relative_path,
                    'progress': download_progress.get(relative_path, 100 if os.path.exists(os.path.join(models_path, relative_path)) else 0)
                })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return jsonify(models)

@app.route('/api/download-model', methods=['POST'])
def download_model():
    model_repo = request.json.get('model_repo')
    if not model_repo:
        return jsonify({'error': 'Model repository not provided'}), 400

    pvc_mount_path = f"/mnt/models/{model_repo}"
    repo_url = f"https://huggingface.co/{model_repo}"

    def download_process():
        try:
            if not os.path.exists(pvc_mount_path):
                socketio.emit('download_status', {'message': f"Creating directory {pvc_mount_path}"})
                os.makedirs(pvc_mount_path)

            if not os.listdir(pvc_mount_path):
                socketio.emit('download_status', {'message': f"Cloning repository {repo_url} into {pvc_mount_path}"})
                process = subprocess.Popen(
                    ["git", "clone", repo_url, pvc_mount_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )

                total_objects = None
                received_objects = 0

                for line in process.stdout:
                    if "remote: Counting objects:" in line:
                        total_objects = int(line.split()[3])
                    elif "Receiving objects:" in line and "%" in line:
                        received_objects = int(line.split()[2].split('/')[0])
                        progress = int((received_objects / total_objects) * 100) if total_objects else 0
                        download_progress[model_repo] = progress
                        socketio.emit('download_progress', {'model': model_repo, 'progress': progress})

                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, process.args)

                download_progress[model_repo] = 100
                socketio.emit('download_progress', {'model': model_repo, 'progress': 100})
                socketio.emit('download_status', {'message': "Model repository successfully cloned."})
            else:
                socketio.emit('download_status', {'message': "Repository already exists in PVC, skipping clone."})

            socketio.emit('download_status', {'message': "Download process completed.", 'complete': True})
        except subprocess.CalledProcessError as e:
            socketio.emit('download_status', {'message': f"Error during git operation: {str(e)}", 'error': True})
        except Exception as e:
            socketio.emit('download_status', {'message': f"An error occurred: {str(e)}", 'error': True})

    thread = threading.Thread(target=download_process)
    thread.start()

    return jsonify({'message': 'Download process started'})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port='8080')