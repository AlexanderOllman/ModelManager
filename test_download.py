from huggingface_hub import snapshot_download

# Specify the local cache directory and the model name
cache_dir = "/mnt/models-pvc/"
model_name = "roneneldan/TinyStories-1M"

# Download the model files to the cache directory without loading them into memory
snapshot_download(
    repo_id=model_name, 
    cache_dir=cache_dir,
    local_files_only=False,
    allow_patterns=["*.safetensors", "*.json"]  # This downloads only safetensors and json files
)

print(f"Model files downloaded to {cache_dir}")