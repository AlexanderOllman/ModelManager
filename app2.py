from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO
import subprocess
import json
import yaml
import threading
import time
import os
import logging
from huggingface_hub import snapshot_download, logging as hf_logging
import contextlib
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
hf_logging.set_verbosity_info()

app = Flask(__name__)
socketio = SocketIO(app)

deployment_lock = threading.Lock()
deployment_in_progress = False

def run_kubectl_command(command):
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    return result.stdout, result.stderr

def ensure_model_directory():
    models_path = "/mnt/models/hub"
    try:
        os.makedirs(models_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating models directory: {e}")
        return False

def check_deployment_status(namespace, model_name, framework='nvidia-nim'):
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
                        return True

        attempt += 1
        time.sleep(10)

    socketio.emit('deployment_status', {
        'status': 'error',
        'message': "Deployment timed out",
        'final': True
    })
    return False

def generate_vllm_manifest(data):
    manifest = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": f"vllm-{data['modelName']}",
            "annotations": {
                "autoscaling.knative.dev/target": "1",
                "autoscaling.knative.dev/class": "kpa.autoscaling.knative.dev",
                "autoscaling.knative.dev/minScale": "1",
                "autoscaling.knative.dev/maxScale": "1",
                "serving.knative.dev/scale-to-zero-grace-period": "infinite",
                "serving.kserve.io/enable-auth": "false",
                "serving.knative.dev/scaleToZeroPodRetention": "false"
            }
        },
        "spec": {
            "predictor": {
                "containers": [{
                    "name": "kserve-container",
                    "args": [
                        "--port",
                        "8080",
                        "--model",
                        data['model']
                    ],
                    "command": [
                        "python3",
                        "-m",
                        "vllm.entrypoints.api_server"
                    ],
                    "env": [
                        {"name": "HF_HOME", "value": "/mnt/models"},
                        {"name": "HF_HUB_CACHE", "value": "/mnt/models/hub"},
                        {"name": "XDG_CACHE_HOME", "value": "/mnt/models/.cache"},
                        {"name": "XDG_CONFIG_HOME", "value": "/mnt/models/.config"},
                        {"name": "PROTOCOL", "value": "v2"},
                        {"name": "SCALE_TO_ZERO_ENABLED", "value": "false"}
                    ],
                    "image": data['containerImage'],
                    "imagePullPolicy": "IfNotPresent",
                    "ports": [{
                        "containerPort": 8080,
                        "protocol": "TCP"
                    }],
                    "readinessProbe": {
                        "httpGet": {
                            "path": "/health",
                            "port": 8080
                        },
                        "initialDelaySeconds": 60,
                        "periodSeconds": 10,
                        "timeoutSeconds": 5
                    },
                    "livenessProbe": {
                        "httpGet": {
                            "path": "/health",
                            "port": 8080
                        },
                        "initialDelaySeconds": 60,
                        "periodSeconds": 10,
                        "timeoutSeconds": 5
                    },
                    "resources": {
                        "limits": {
                            "cpu": str(data['resources']['cpu']),
                            "memory": str(data['resources']['memory']),
                            "nvidia.com/gpu": str(data['resources']['gpu'])
                        },
                        "requests": {
                            "cpu": str(data['resources']['cpu']),
                            "memory": str(data['resources']['memory']),
                            "nvidia.com/gpu": str(data['resources']['gpu'])
                        }
                    },
                    "volumeMounts": [{
                        "name": "model-pvc",
                        "mountPath": "/mnt/models"
                    }]
                }],
                "volumes": [{
                    "name": "model-pvc",
                    "persistentVolumeClaim": {
                        "claimName": data['storageUri']
                    }
                }]
            }
        }
    }
    return manifest

def generate_nvidia_manifest(data):
    inference_yaml = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": data['modelName'],
            "annotations": {
                "autoscaling.knative.dev/target": "10"
            }
        },
        "spec": {
            "predictor": {
                "minReplicas": 1,
                "model": {
                    "modelFormat": {
                        "name": f"nvidia-nim-{data['modelName']}"
                    },
                    "resources": {
                        "limits": {
                            "cpu": str(data['resources']['cpu']),
                            "memory": str(data['resources']['memory']),
                            "nvidia.com/gpu": str(data['resources']['gpu'])
                        },
                        "requests": {
                            "cpu": str(data['resources']['cpu']),
                            "memory": str(data['resources']['memory']),
                            "nvidia.com/gpu": str(data['resources']['gpu'])
                        }
                    },
                    "runtime": f"nvidia-nim-{data['modelName']}-runtime",
                    "storageUri": data['storageUri']
                }
            }
        }
    }

    runtime_yaml = {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ClusterServingRuntime",
        "metadata": {
            "name": f"nvidia-nim-{data['modelName']}-runtime"
        },
        "spec": {
            "annotations": {
                "prometheus.kserve.io/path": "/metrics",
                "prometheus.kserve.io/port": "8000",
                "serving.kserve.io/enable-metric-aggregation": "true",
                "serving.kserve.io/enable-prometheus-scraping": "true"
            },
            "containers": [{
                "args": None,
                "env": [{
                    "name": "NIM_CACHE_PATH",
                    "value": "/mnt/models"
                }],
                "image": data['containerImage'],
                "name": "kserve-container",
                "resources": {
                    "limits": {
                        "cpu": str(data['resources']['cpu']),
                        "memory": str(data['resources']['memory'])
                    },
                    "requests": {
                        "cpu": str(int(float(data['resources']['cpu']) // 2)),
                        "memory": str(data['resources']['memory'])
                    }
                }
            }],
            "supportedModelFormats": [{
                "autoSelect": True,
                "name": f"nvidia-nim-{data['modelName']}",
                "version": "1"
            }]
        }
    }

    return inference_yaml, runtime_yaml

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

@app.route('/api/namespaces')
def get_namespaces():
    namespaces_data, _ = run_kubectl_command("kubectl get namespaces -o json")
    namespaces_json = json.loads(namespaces_data)
    return jsonify([item['metadata']['name'] for item in namespaces_json['items']])

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
        
        inference_yaml, runtime_yaml = generate_nvidia_manifest(data)
        
        # Save YAML files
        with open('inference.yaml', 'w') as f:
            yaml.dump(inference_yaml, f)
        with open('runtime.yaml', 'w') as f:
            yaml.dump(runtime_yaml, f)

        # Apply runtime YAML
        socketio.emit('deployment_status', {
            'status': 'info',
            'message': "Deploying runtime..."
        })
        runtime_result, runtime_error = run_kubectl_command(f"kubectl apply -f runtime.yaml")
        if runtime_error:
            socketio.emit('deployment_status', {
                'status': 'error',
                'message': f"Failed to deploy runtime: {runtime_error}",
                'final': True
            })
            deployment_in_progress = False
            return jsonify({'status': 'error', 'message': 'Runtime deployment failed.'})

        # Wait for runtime to be ready
        time.sleep(15)

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

        # If this is a custom model (not already downloaded)
        if 'customModel' in data:
            socketio.emit('deployment_status', {
                'status': 'info',
                'message': f"Downloading model {data['model']} from HuggingFace..."
            })
            try:
                result = snapshot_download(
                    repo_id=data['model'],
                    cache_dir="/mnt/models/hub",
                    local_files_only=False,
                    allow_patterns=["*.safetensors", "*.json"]
                )
                socketio.emit('deployment_status', {
                    'status': 'info',
                    'message': f"Model downloaded successfully to {result}"
                })
            except Exception as e:
                socketio.emit('deployment_status', {
                    'status': 'error',
                    'message': f"Failed to download model: {str(e)}",
                    'final': True
                })
                deployment_in_progress = False
                return jsonify({'status': 'error', 'message': 'Model download failed'})

        # Generate and apply vLLM manifest
        manifest = generate_vllm_manifest(data)
        
        # Save manifest
        with open('vllm.yaml', 'w') as f:
            yaml.dump(manifest, f)

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
        return Response("data: Error: No model repository specified\n\n", 
                       mimetype='text/event-stream')

    if not ensure_model_directory():
        return Response("data: Error: Could not create models directory\n\n", 
                       mimetype='text/event-stream')

    @stream_with_context
    def generate():
        try:
            cache_dir = "/mnt/models/hub"
            yield f"data: Starting download of {model_repo}\n\n"
            
            result = snapshot_download(
                repo_id=model_repo,
                cache_dir=cache_dir,
                local_files_only=False,
                allow_patterns=["*.safetensors", "*.json"]
            )
            
            yield f"data: Model files downloaded to {cache_dir}\n\n"
            yield f"data: Downloaded files located at: {result}\n\n"
            
            # List downloaded files
            try:
                files = os.listdir(result)
                yield f"data: Downloaded files: {', '.join(files)}\n\n"
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

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port="8080")
    