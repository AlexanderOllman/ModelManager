from flask import Flask, render_template, jsonify, request
import subprocess
import json
import os
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream

app = Flask(__name__)

def load_kubernetes_config():
    try:
        config.load_incluster_config()
        print("Loaded in-cluster configuration")
    except config.config_exception.ConfigException:
        try:
            config.load_kube_config()
            print("Loaded local kubeconfig")
        except config.config_exception.ConfigException:
            print("Failed to load both in-cluster and local configurations")
            return None
    
    return client.CoreV1Api()

v1 = load_kubernetes_config()

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
    if not v1:
        return jsonify({"error": "Kubernetes configuration not available"}), 500

    pvc_name = request.args.get('pvc')
    namespace = request.args.get('namespace', 'default')

    if not pvc_name:
        return jsonify({"error": "PVC name is required"}), 400

    try:
        # Get the PVC
        pvc = v1.read_namespaced_persistent_volume_claim(pvc_name, namespace)

        # Create a temporary pod to mount the PVC
        pod_name = f"pvc-reader-{pvc_name.lower()}"
        pod_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": pod_name},
            "spec": {
                "containers": [{
                    "name": "pvc-reader",
                    "image": "busybox",
                    "command": ["sleep", "3600"],
                    "volumeMounts": [{
                        "name": "pvc-mount",
                        "mountPath": "/mnt/pvc"
                    }]
                }],
                "volumes": [{
                    "name": "pvc-mount",
                    "persistentVolumeClaim": {"claimName": pvc_name}
                }],
                "restartPolicy": "Never"
            }
        }

        # Create the pod
        v1.create_namespaced_pod(namespace, pod_manifest)

        # Wait for the pod to be ready
        while True:
            pod = v1.read_namespaced_pod(pod_name, namespace)
            if pod.status.phase == 'Running':
                break

        # Execute 'ls -lh' command in the pod
        exec_command = ['ls', '-lh', '/mnt/pvc']
        resp = stream(v1.connect_get_namespaced_pod_exec,
                      pod_name,
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

        # Delete the temporary pod
        v1.delete_namespaced_pod(pod_name, namespace)

        return jsonify(files)

    except ApiException as e:
        return jsonify({"error": f"Kubernetes API error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8080')