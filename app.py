from flask import Flask, render_template, jsonify, request
import subprocess
import json
import yaml

app = Flask(__name__)

def run_kubectl_command(command):
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    return result.stdout, result.stderr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models')
def models():
    isvc_data = run_kubectl_command("kubectl get isvc -A -o json")[0]
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
    isvc_data = run_kubectl_command("kubectl get isvc -A -o json")[0]
    return jsonify(json.loads(isvc_data))

@app.route('/api/clusterservingruntime')
def get_clusterservingruntime():
    csr_data = run_kubectl_command("kubectl get clusterservingruntime -A -o json")[0]
    return jsonify(json.loads(csr_data))

@app.route('/api/deploy', methods=['POST'])
def deploy_model():
    data = request.json
    runtime_name = data['runtime']
    namespace = data['namespace']

    # Check if runtime exists
    runtime_check = run_kubectl_command(f"kubectl get clusterservingruntime {runtime_name}")[1]
    runtime_exists = "NotFound" not in runtime_check

    if runtime_exists:
        return jsonify({
            'runtimeExists': True,
            'deploymentData': data
        })
    else:
        return deploy_without_runtime_update(data)

@app.route('/api/deploy/update-runtime', methods=['POST'])
def deploy_with_runtime_update():
    data = request.json
    runtime_name = data['runtime']
    namespace = data['namespace']

    # Delete existing runtime
    run_kubectl_command(f"kubectl delete clusterservingruntime {runtime_name}")
    
    # Deploy updated runtime and inference service
    return deploy_without_runtime_update(data, update=True)

@app.route('/api/deploy/without-runtime-update', methods=['POST'])
def deploy_without_runtime_update(data=None, update=False):
    if data is None:
        data = request.json

    inference_yaml = yaml.safe_load(data['inferenceYaml'])
    runtime_yaml = yaml.safe_load(data['runtimeYaml'])
    namespace = data['namespace']

    # Save YAML files
    with open('inference.yaml', 'w') as f:
        yaml.dump(inference_yaml, f)
    with open('runtime.yaml', 'w') as f:
        yaml.dump(runtime_yaml, f)

    if update:
        # Apply runtime YAML
        runtime_result, runtime_error = run_kubectl_command(f"kubectl apply -f runtime.yaml")

        if runtime_error:
            return jsonify({
                'success': False,
                'error': f"Failed to deploy runtime: {runtime_error}"
            })
    else:
        runtime_result = "Update not needed."

    # Apply inference YAML
    
    inference_result, inference_error = run_kubectl_command(f"kubectl apply -f inference.yaml -n {namespace}")

    if inference_error:
        return jsonify({
            'success': False,
            'error': f"Failed to deploy inference service: {inference_error}"
        })

    return jsonify({
        'success': True,
        'runtime_result': runtime_result,
        'inference_result': inference_result
    })

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
    app.run(debug=True, host='0.0.0.0', port='8080')