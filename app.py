from flask import Flask, render_template, jsonify, request, Response, stream_with_context
import subprocess
import json
import yaml
import threading
import time
import os
import logging
from flask_socketio import SocketIO
from huggingface_hub import snapshot_download, logging as hf_logging, HfApi
import sys
import contextlib
import io

app = Flask(__name__)
socketio = SocketIO(app)

deployment_lock = threading.Lock()
deployment_in_progress = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set HuggingFace Hub logging to INFO level
hf_logging.set_verbosity_info()

class HFLoggingHandler(logging.Handler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        log_entry = self.format(record)
        self.callback(log_entry)

class StringIOWithCallback(io.StringIO):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def write(self, s):
        super().write(s)
        if s.strip():  # Only send non-empty lines
            self.callback(s.strip())
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
            logger.warning(f"Models path {models_path} does not exist")
            return jsonify([])
        
        model_info = []
        # Only get immediate subdirectories
        for item in os.listdir(models_path):
            full_path = os.path.join(models_path, item)
            if os.path.isdir(full_path):
                # Check if directory follows the expected format
                if item.startswith("models--"):
                    try:
                        # Split the directory name into parts
                        parts = item.split("--")
                        if len(parts) >= 3:  # models--owner--repo
                            repo_owner = parts[1]
                            repo_name = "--".join(parts[2:])  # Join remaining parts in case repo name contains --
                            model_info.append({
                                "directory": item,
                                "repo_owner": repo_owner,
                                "repo_name": repo_name,
                                "full_repo": f"{repo_owner}/{repo_name}"
                            })
                    except Exception as e:
                        logger.error(f"Error parsing directory name {item}: {e}")
                        continue
        
        logger.info(f"Found {len(model_info)} models")
        return jsonify(model_info)
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify([])

def ensure_model_directory():
    models_path = "/mnt/models/hub"
    try:
        logger.info(f"Checking if models directory exists: {models_path}")
        if os.path.exists(models_path):
            logger.info("Models directory already exists")
        else:
            logger.info("Creating models directory")
            os.makedirs(models_path, exist_ok=True)
            logger.info("Models directory created successfully")

        # Verify write permissions
        test_file = os.path.join(models_path, '.write_test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info("Write permission test passed")
        except Exception as e:
            logger.error(f"Write permission test failed: {e}")
            return False

        return True
    except Exception as e:
        logger.error(f"Error ensuring models directory exists: {e}")
        return False

@app.route('/api/download-model')
def download_model():
    model_repo = request.args.get('repo')
    if not model_repo:
        logger.error("No model repository specified")
        return Response("data: Error: No model repository specified\n\n", 
                       mimetype='text/event-stream')

    if not ensure_model_directory():
        logger.error("Could not create models directory")
        return Response("data: Error: Could not create models directory\n\n", 
                       mimetype='text/event-stream')

    @stream_with_context
    def generate():
        def send_log(message):
            yield f"data: {message}\n\n"

        try:
            cache_dir = "/mnt/models/hub"
            logger.info(f"Starting download of {model_repo}")
            yield f"data: Starting download of {model_repo}\n\n"

            # Create a callback for logging
            def log_callback(message):
                logger.info(message)
                yield f"data: {message}\n\n"

            # Set up HuggingFace logging handler
            hf_handler = HFLoggingHandler(lambda msg: next(send_log(msg)))
            hf_logger = logging.getLogger("huggingface_hub")
            hf_logger.addHandler(hf_handler)

            # Get model info
            api = HfApi()
            try:
                logger.info(f"Fetching model info for {model_repo}")
                yield f"data: Fetching model info for {model_repo}\n\n"
                model_info = api.model_info(model_repo)
                yield f"data: Model size: {model_info.siblings_size_human_readable}\n\n"
            except Exception as e:
                logger.warning(f"Could not fetch model info: {e}")
                yield f"data: Could not fetch model size information\n\n"

            # Capture stdout and stderr
            stdout_callback = StringIOWithCallback(lambda msg: next(send_log(f"stdout: {msg}")))
            stderr_callback = StringIOWithCallback(lambda msg: next(send_log(f"stderr: {msg}")))

            logger.info("Setting up download parameters")
            yield f"data: Setting up download parameters\n\n"

            with contextlib.redirect_stdout(stdout_callback), contextlib.redirect_stderr(stderr_callback):
                logger.info("Starting snapshot_download")
                yield f"data: Starting snapshot_download\n\n"
                
                result = snapshot_download(
                    repo_id=model_repo,
                    cache_dir=cache_dir,
                    local_files_only=False,
                    allow_patterns=["*.safetensors", "*.json"],
                    token=None  # Add your token here if needed for private repos
                )
                
                logger.info(f"Download completed. Files stored at: {result}")
                yield f"data: Model files downloaded to {cache_dir}\n\n"
                yield f"data: Downloaded files located at: {result}\n\n"
                
                # List downloaded files
                try:
                    files = os.listdir(result)
                    logger.info(f"Downloaded files: {files}")
                    yield f"data: Downloaded files: {', '.join(files)}\n\n"
                    
                    # Calculate total size of downloaded files
                    total_size = sum(os.path.getsize(os.path.join(result, f)) for f in files)
                    size_mb = total_size / (1024 * 1024)
                    logger.info(f"Total downloaded size: {size_mb:.2f} MB")
                    yield f"data: Total downloaded size: {size_mb:.2f} MB\n\n"
                except Exception as e:
                    logger.error(f"Error listing downloaded files: {e}")
                    yield f"data: Error listing downloaded files: {str(e)}\n\n"

            yield "data: Download process complete!\n\n"
            logger.info("Download process complete")

        except Exception as e:
            error_msg = f"Error downloading model: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield f"data: {error_msg}\n\n"
            # Log the full traceback
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Full traceback:\n{tb}")
            yield f"data: Full error details: {tb}\n\n"
        finally:
            # Clean up the HuggingFace logging handler
            hf_logger.removeHandler(hf_handler)

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port='8080')