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
import shutil
from pathlib import Path

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


def check_deployment_status(namespace, model_name, framework='nvidia-nim'):
    global deployment_in_progress
    max_attempts = 30
    attempt = 0
    while attempt < max_attempts:
        # Check pod status
        pod_command = f"kubectl get pods -n {namespace} -l serving.kserve.io/inferenceservice={model_name} -o json"
        pod_output, pod_error = run_kubectl_command(pod_command)
        if pod_error:
            socketio.emit('deployment_status', {
                'status': 'error', 
                'message': f"Error checking pod status: {pod_error}"
            })
            deployment_in_progress = False
            return False

        pod_data = json.loads(pod_output)
        if pod_data['items']:
            pod_status = pod_data['items'][0]['status']['phase']
            socketio.emit('deployment_status', {
                'status': 'info',
                'message': f"Pod status: {pod_status}"
            })
            
            if pod_status == 'Running':
                # Check InferenceService status
                isvc_command = f"kubectl get inferenceservice {model_name} -n {namespace} -o json"
                isvc_output, isvc_error = run_kubectl_command(isvc_command)
                if isvc_error:
                    socketio.emit('deployment_status', {
                        'status': 'error',
                        'message': f"Error checking InferenceService status: {isvc_error}"
                    })
                    deployment_in_progress = False
                    return False

                isvc_data = json.loads(isvc_output)
                conditions = isvc_data.get('status', {}).get('conditions', [])
                ready_condition = next((c for c in conditions if c['type'] == 'Ready'), None)
                
                if ready_condition:
                    socketio.emit('deployment_status', {
                        'status': 'info',
                        'message': f"InferenceService status: {ready_condition['status']}"
                    })
                    
                    if ready_condition['status'] == 'True':
                        url = isvc_data.get('status', {}).get('url', 'N/A')
                        socketio.emit('deployment_status', {
                            'status': 'success',
                            'message': f"Model deployed successfully! URL: {url}",
                            'final': True
                        })
                        deployment_in_progress = False
                        return True

        attempt += 1
        time.sleep(10)  # Wait 10 seconds between checks

    socketio.emit('deployment_status', {
        'status': 'error',
        'message': "Deployment timed out",
        'final': True
    })
    deployment_in_progress = False
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/manage')
def manage():
    return render_template('manage.html')

@app.route('/deploy')
def deploy():
    return render_template('deploy.html')

@app.route('/download')
def download():
    return render_template('download.html')

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
                            repo_name = "--".join(parts[2:])
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


def validate_manifest_data(data, required_fields):
    """Validate manifest input data."""
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")


def validate_resources(resources):
    """Validate and format resource requirements."""
    required_fields = ['cpu', 'memory', 'gpu']
    if not all(field in resources for field in required_fields):
        raise ValueError("Missing required resource fields. Must specify cpu, memory, and gpu.")
    
    try:
        # Format CPU and Memory, ensure GPU is an integer
        cpu = str(resources['cpu'])
        memory = resources['memory'] if resources['memory'].endswith('Gi') else f"{resources['memory']}Gi"
        gpu = str(resources['gpu']) if isinstance(resources['gpu'], int) else resources['gpu']
        
        return {
            'cpu': cpu,
            'memory': memory,
            'nvidia.com/gpu': gpu
        }
    except Exception as e:
        raise ValueError(f"Invalid resource format: {str(e)}")

def calculate_request_cpu(limits_cpu_str):
    """Calculate request CPU from limits CPU."""
    limits_cpu = float(limits_cpu_str)
    request_cpu = max(1, int(limits_cpu // 2))
    return str(request_cpu)

def generate_vllm_manifest(data):
    """Generate vLLM manifest using string templates with correct formatting and explicit quotations."""
    required_fields = ['modelName', 'model', 'containerImage', 'resources', 'storageUri']
    validate_manifest_data(data, required_fields)
    
    try:
        resources = validate_resources(data['resources'])
        request_cpu = calculate_request_cpu(resources['cpu'])
        
        # Prepare the manifest template
        manifest_template = f"""
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: vllm-{data['modelName']}
  annotations:
    autoscaling.knative.dev/target: "1"
    autoscaling.knative.dev/class: "kpa.autoscaling.knative.dev"
    autoscaling.knative.dev/minScale: "1"
    autoscaling.knative.dev/maxScale: "1"
    serving.knative.dev/scale-to-zero-grace-period: "infinite"
    serving.kserve.io/enable-auth: "false"
    serving.knative.dev/scaleToZeroPodRetention: "false"
spec:
  predictor:
    containers:
      - name: "kserve-container"
        args:
          - "--port"
          - "8080"
          - "--model"
          - "{data['model']}"
        command:
          - "python3"
          - "-m"
          - "vllm.entrypoints.api_server"
        env:
          - name: "HF_HOME"
            value: "/mnt/models-pvc"
          - name: "HF_HUB_CACHE"
            value: "/mnt/models-pvc/hub"
          - name: "XDG_CACHE_HOME"
            value: "/mnt/models-pvc/.cache"
          - name: "XDG_CONFIG_HOME"
            value: "/mnt/models-pvc/.config"
          - name: "PROTOCOL"
            value: "v2"
          - name: "SCALE_TO_ZERO_ENABLED"
            value: "false"
        image: "{data['containerImage']}"
        imagePullPolicy: "IfNotPresent"
        ports:
          - containerPort: 8080
            protocol: "TCP"
        readinessProbe:
          httpGet:
            path: "/health"
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 10
          timeoutSeconds: 5
        livenessProbe:
          httpGet:
            path: "/health"
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 10
          timeoutSeconds: 5
        resources:
          limits:
            cpu: "{resources['cpu']}"
            memory: "{resources['memory']}"
            nvidia.com/gpu: "{resources['nvidia.com/gpu']}"
          requests:
            cpu: "{request_cpu}"
            memory: "{resources['memory']}"
            nvidia.com/gpu: "{resources['nvidia.com/gpu']}"
        volumeMounts:
          - name: "model-pvc"
            mountPath: "/mnt/models-pvc"
    volumes:
      - name: "model-pvc"
        persistentVolumeClaim:
          claimName: "{data['storageUri']}"
"""
        return manifest_template.strip()
    
    except Exception as e:
        raise ValueError(f"Error generating vLLM manifest: {str(e)}")

def generate_nvidia_inferenceservice_manifest(data):
    """Generate NVIDIA InferenceService manifest using string templates for correct formatting."""
    required_fields = ['modelName', 'containerImage', 'resources', 'storageUri']
    validate_manifest_data(data, required_fields)

    try:
        resources = validate_resources(data['resources'])
        request_cpu = calculate_request_cpu(resources['cpu'])
        model_name = data['modelName']

        # Prepare the inference service manifest template
        inference_yaml = f"""
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: {model_name}
  annotations:
    autoscaling.knative.dev/target: "1"
    autoscaling.knative.dev/class: "kpa.autoscaling.knative.dev"
    autoscaling.knative.dev/minScale: "1"
    autoscaling.knative.dev/maxScale: "1"
    serving.knative.dev/scale-to-zero-grace-period: "infinite"
    serving.kserve.io/enable-auth: "false"
    serving.knative.dev/scaleToZeroPodRetention: "false"
spec:
  predictor:
    containers:
      name: kserve-container
      image: {data['containerImage']}
      env:
        - name: NIM_CACHE_PATH
          value: /mnt/models-pvc
      resources:
        limits:
          cpu: "{resources['cpu']}"
          memory: "{resources['memory']}"
          nvidia.com/gpu: "{resources['nvidia.com/gpu']}"
        requests:
          cpu: "{request_cpu}"
          memory: "{resources['memory']}"
          nvidia.com/gpu: "{resources['nvidia.com/gpu']}"
    model:
      modelFormat:
        name: nvidia-nim-{model_name}
      storageUri: {data['storageUri']}
"""

        return inference_yaml.strip()
    except Exception as e:
        raise ValueError(f"Error generating NVIDIA InferenceService manifest: {str(e)}")



def save_manifest_to_file(manifest, filename):
    """Save manifest to a YAML file with correct formatting."""
    with open(filename, 'w') as f:
        yaml.safe_dump(manifest, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

# Update the deployment routes to use the new YAML dumper
@app.route('/api/deploy-vllm', methods=['POST'])
def deploy_vllm():
    global deployment_in_progress
    
    if deployment_in_progress:
        return jsonify({
            'status': 'error',
            'message': 'A deployment is already in progress. Please wait.'
        })
    
    with deployment_lock:
        if deployment_in_progress:
            return jsonify({
                'status': 'error',
                'message': 'A deployment is already in progress. Please wait.'
            })
        deployment_in_progress = True
    
    try:
        data = request.json
        namespace = data['namespace']
        model_name = data['modelName']

        socketio.emit('deployment_status', {
            'status': 'info',
            'message': "Starting vLLM deployment process..."
        })

        # Generate and apply vLLM manifest
        manifest = generate_vllm_manifest(data)
        
        # Save manifest with proper formatting
        with open('vllm.yaml', 'w') as f:
            f.write(manifest)

        # Apply manifest
        socketio.emit('deployment_status', {
            'status': 'info',
            'message': "Deploying vLLM service..."
        })
        
        result, error = run_kubectl_command(f"kubectl apply -f vllm.yaml -n {namespace}")
        if error:
            socketio.emit('deployment_status', {
                'status': 'error',
                'message': f"Failed to deploy vLLM service: {error}",
                'final': True
            })
            deployment_in_progress = False
            return jsonify({'status': 'error', 'message': 'vLLM service deployment failed'})

        # Start status checking in a separate thread
        threading.Thread(
            target=lambda: check_deployment_status(
                namespace, 
                f"vllm-{model_name}", 
                framework='vllm'
            )
        ).start()

        return jsonify({'status': 'info', 'message': 'Deployment started. Checking status...'})

    except Exception as e:
        logger.error(f"Error in vLLM deployment: {str(e)}")
        socketio.emit('deployment_status', {
            'status': 'error',
            'message': f"Deployment error: {str(e)}",
            'final': True
        })
        deployment_in_progress = False
        return jsonify({'status': 'error', 'message': f'Deployment error: {str(e)}'})


@app.route('/api/deploy', methods=['POST'])
def deploy_model():
    global deployment_in_progress
    
    if deployment_in_progress:
        return jsonify({
            'status': 'error',
            'message': 'A deployment is already in progress. Please wait.'
        })
    
    with deployment_lock:
        if deployment_in_progress:
            return jsonify({
                'status': 'error',
                'message': 'A deployment is already in progress. Please wait.'
            })
        deployment_in_progress = True
    
    try:
        data = request.json
        namespace = data['namespace']
        model_name = data['modelName']
        
        socketio.emit('deployment_status', {
            'status': 'info',
            'message': "Starting deployment process..."
        })
        
        inference_yaml = generate_nvidia_inferenceservice_manifest(data)
        
        with open('inference.yaml', 'w') as f:
            f.write(inference_yaml)
        
        # Apply inference YAML
        socketio.emit('deployment_status', {
            'status': 'info',
            'message': "Deploying inference service..."
        })
        inference_result, inference_error = run_kubectl_command(
            f"kubectl apply -f inference.yaml -n {namespace}"
        )
        if inference_error:
            socketio.emit('deployment_status', {
                'status': 'error',
                'message': f"Failed to deploy inference service: {inference_error}",
                'final': True
            })
            deployment_in_progress = False
            return jsonify({'status': 'error', 'message': 'Inference service deployment failed.'})
        
        # Start status checking in a separate thread
        threading.Thread(
            target=lambda: check_deployment_status(namespace, model_name)
        ).start()

        return jsonify({'status': 'info', 'message': 'Deployment started. Checking status...'})

    except Exception as e:
        logger.error(f"Error in deployment: {str(e)}")
        socketio.emit('deployment_status', {
            'status': 'error',
            'message': f"Deployment error: {str(e)}",
            'final': True
        })
        deployment_in_progress = False
        return jsonify({'status': 'error', 'message': f'Deployment error: {str(e)}'})
    finally:
        deployment_in_progress = False

@app.route('/api/pod-logs/<namespace>/<pod_name>')
def get_pod_logs(namespace, pod_name):
    command = f"kubectl logs -n {namespace} {pod_name}"
    logs, error = run_kubectl_command(command)
    if error:
        return str(error), 500
    return logs

@app.route('/api/describe/<resource_type>/<resource_name>')
def describe_resource(resource_type, resource_name):
    namespace = request.args.get('namespace')
    
    resource_map = {
        'isvc': 'inferenceservice',
        'csr': 'clusterservingruntime'
    }
    
    resource = resource_map.get(resource_type)
    if not resource:
        return jsonify({'error': 'Invalid resource type'}), 400

    command = f"kubectl describe {resource} {resource_name}"
    if namespace:
        command += f" -n {namespace}"
        
    description, error = run_kubectl_command(command)
    if error:
        return jsonify({'error': str(error)}), 500
        
    return jsonify({'description': description})


@app.route('/api/delete-model-files', methods=['DELETE'])
def delete_model_files():
    try:
        repo = request.args.get('repo')
        if not repo:
            return jsonify({'success': False, 'error': 'No repository specified'}), 400

        # Convert repo/model format to directory format
        # e.g., 'roneneldan/TinyStories-1M' -> 'models--roneneldan--TinyStories-1M'
        repo_parts = repo.split('/')
        if len(repo_parts) != 2:
            return jsonify({'success': False, 'error': 'Invalid repository format'}), 400

        dir_name = f"models--{repo_parts[0]}--{repo_parts[1]}"
        
        # Base path for models
        models_base_path = '/mnt/models/hub'
        model_path = os.path.join(models_base_path, dir_name)

        if not os.path.exists(model_path):
            return jsonify({'success': False, 'error': 'Model directory not found'}), 404

        # Safety check to ensure we're only deleting within the models directory
        if not Path(model_path).resolve().is_relative_to(Path(models_base_path).resolve()):
            return jsonify({'success': False, 'error': 'Invalid model path'}), 400

        # Delete the directory and all its contents
        shutil.rmtree(model_path)

        # Log the deletion
        app.logger.info(f"Deleted model directory: {dir_name}")

        return jsonify({
            'success': True,
            'message': f'Successfully deleted model files for {repo}'
        })

    except Exception as e:
        app.logger.error(f"Error deleting model files: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error deleting model files: {str(e)}'
        }), 500
    
@app.route('/api/delete/<resource_type>/<resource_name>', methods=['DELETE'])
def delete_resource(resource_type, resource_name):
    namespace = request.args.get('namespace')
    
    resource_map = {
        'isvc': 'inferenceservice',
        'csr': 'clusterservingruntime'
    }
    
    resource = resource_map.get(resource_type)
    if not resource:
        return jsonify({'success': False, 'error': 'Invalid resource type'}), 400

    command = f"kubectl delete {resource} {resource_name}"
    if namespace:
        command += f" -n {namespace}"
        
    output, error = run_kubectl_command(command)
    if error:
        return jsonify({'success': False, 'error': str(error)}), 500
        
    return jsonify({
        'success': True, 
        'message': f'{resource_type.upper()} {resource_name} deleted successfully'
    })

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
# Example usage
    data = {
        'modelName': 'facebook-opt-125m',
        'model': 'facebook/opt-125m',
        'containerImage': 'vllm/vllm-openai:latest',
        'resources': {'cpu': 4, 'memory': '8Gi', 'gpu': 1},
        'storageUri': 'models-pvc'
    }

    vllm_manifest = generate_vllm_manifest(data)
    with open('vllm.yaml', 'w') as f:
        f.write(vllm_manifest)

    inference_yaml = generate_nvidia_inferenceservice_manifest(data)
    with open('inference.yaml', 'w') as f:
        f.write(inference_yaml)

    socketio.run(app, debug=True, host='0.0.0.0', port='8080')
    