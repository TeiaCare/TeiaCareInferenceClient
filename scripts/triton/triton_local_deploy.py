import os
import logging
import docker
import argparse

import threading

def log_container_output(container):
    for log_entry in container.logs(stream=True, follow=True):
        log_entry = log_entry.decode('utf-8').strip()
        print(log_entry)

def has_gpu():
    nvidia_gpu_path = '/proc/driver/nvidia/gpus'
    
    logger = logging.getLogger(__name__)

    if os.path.exists(nvidia_gpu_path):
        logger.info("NVIDIA GPU exists!")
        return True
    else:
        logger.info("NVIDIA GPU does not exist.")
        return False
    

def start_docker_compose(compose_file_path):
    client = docker.from_env()
    compose = docker.types.ComposeFile(compose_file_path)
    services = compose.services()
    client.api.start(compose.services())

def stop_docker_compose(compose_file_path):
    client = docker.from_env()
    compose = docker.types.ComposeFile(compose_file_path)
    services = compose.services()
    client.api.stop(compose.services())

def run_triton_server(image_name, path_to_host_models, env=None):
    client = docker.from_env()
    container = client.containers.run(
        image=image_name,
        detach=True,  # Run the container in the background
        remove=True,  # Remove the container when it stops
        ports={'8000': 8000, '8001': 8001, '8002': 8002},
        volumes={path_to_host_models: {'bind': '/models', 'mode': 'rw'}},
        environment=env,
        command='tritonserver --model-repository=/models --strict-model-config=false --log-verbose=true'
    )

    # Create a thread to log the container's output in real-time
    log_thread = threading.Thread(target=log_container_output, args=(container,))
    log_thread.daemon = True  # Exit the thread when the main program exits
    log_thread.start()

    # Wait for the container to stop (you can add more logic here if needed)
    container.wait()

def main(args):
    # Set up logging configuration
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting the script.")

    # Log the argparse arguments
    logger.info(f"Image Name: {args.image_name}")
    logger.info(f"Path to Models Root: {args.path_to_models_root}")
    logger.info(f"Build TRT Models on the Fly: {args.build_trt_models_on_the_fly}")


    gpu_exists = has_gpu()
    environment={'NVIDIA_VISIBLE_DEVICES': 'all'}
    if gpu_exists:
        logger.info("NVIDIA GPU check passed!")
    else:
        logger.info("NVIDIA GPU check failed, going using CPU for inference.")    
        environment={}

    run_triton_server(args.image_name, args.path_to_models_root, environment)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Triton Inference Server local deploy')
    
    # Add arguments to the parser
    parser.add_argument('--image_name', help='Triton Image Name', default='nvcr.io/nvidia/tritonserver:23.04-py3', type=str)
    parser.add_argument('--path_to_models_root', help='Path to models on NAS share', default='/path/to/nas/models/root', type=str)
    parser.add_argument('--build_trt_models_on_the_fly', action='store_true', default=False, help='Build TensorRT models on the fly at the start of Triton Server')
    
    args = parser.parse_args()
    main(args)