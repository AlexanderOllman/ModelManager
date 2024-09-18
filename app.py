from flask import Flask, render_template, jsonify
import subprocess
import json
from kubernetes import client, config

app = Flask(__name__)

# Load Kubernetes configuration
config.load_kube_config()
v1 = client.CoreV1Api()

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

@app.route('/api/pvc-models')
def get_pvc_models():
    pvc_name = "models-pvc"
    namespace = "default"  # Replace with the correct namespace if different

    try:
        # Get the PVC
        pvc = v1.read_namespaced_persistent_volume_claim(pvc_name, namespace)

        # Get the pod that mounts this PVC
        pods = v1.list_namespaced_pod(namespace)
        mounting_pod = None
        for pod in pods.items:
            for volume in pod.spec.volumes:
                if volume.persistent_volume_claim and volume.persistent_volume_claim.claim_name == pvc_name:
                    mounting_pod = pod
                    break
            if mounting_pod:
                break

        if not mounting_pod:
            return jsonify({"error": "No pod found mounting the PVC"}), 404

        # Execute 'ls -lh' command in the pod
        exec_command = ['/bin/sh', '-c', 'ls -lh /mnt/models']  # Adjust the mount path if necessary
        resp = stream(v1.connect_get_namespaced_pod_exec,
                      mounting_pod.metadata.name,
                      namespace,
                      command=exec_command,
                      stderr=True, stdin=False,
                      stdout=True, tty=False)

        # Parse the output
        files = []
        for line in resp.split('\n')[1:]:  # Skip the first line (total size)
            if line.strip():
                parts = line.split()
                if len(parts) >= 8:
                    size, name = parts[4], ' '.join(parts[8:])
                    files.append({"name": name, "size": size})

        return jsonify(files)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def stream(func, *args, **kwargs):
    return func(*args, **kwargs)

if __name__ == '__main__':
    app.run(debug=True)