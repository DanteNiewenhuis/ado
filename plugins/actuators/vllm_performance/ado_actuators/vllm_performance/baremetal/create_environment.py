# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging

from ado_actuators.vllm_performance.k8s.manage_components import (
    ComponentsManager,
)
from ado_actuators.vllm_performance.k8s.yaml_support.build_components import (
    ComponentsYaml,
    VLLMDtype,
)

logger = logging.getLogger(__name__)


def create_test_environment(
        model_name: str,
        base_url: str = "http://localhost:8000",
        tensor_parallel_size: int = 1,
        max_model_len: int = -1,
        max_num_batched_tokens: int = -1,
        max_num_seqs: int = 256,
        hf_token: str | None = None):
    """
    This function serves vLLM with the given configuration.
    This is used for the test-deployment-baremetal-v1 experiment where we want to test the performance of vLLM
    on a baremetal machine without doing any kubernetes deployment. This is useful to isolate the performance of vLLM
    from the performance of the kubernetes deployment.

    The function will:
    1. Check if vLLM is already running with the given configuration. If it is, it will return the URL of the vLLM server.
    2. Serve vLLM with the given configuration if it is not already running, and return the URL of the vLLM server.
    """
    
    import requests
    import uuid
    
    print(f"Start loading the LLM")
    
    log_file_name = f"vllm_serve-{uuid.uuid4()}.log"
    log_file = open(log_file_name, "w")

    logger.debug(f"Starting vLLM server and logging to {log_file_name}...")
    
    env = dict(os.environ)
    env["VLLM_BENCH_LOGLEVEL"] = logging.getLevelName(logger.getEffectiveLevel())

    if hf_token is not None:
        env["HF_TOKEN"] = hf_token

    command = ["vllm", 
               "serve", 
               model_name, 
                "--tensor-parallel-size",
                str(tensor_parallel_size),
                "--max-num-seqs",
                str(max_num_seqs),
               "--host", 
               "0.0.0.0", 
               "--port", 
               "8000"]
    
    if max_model_len > 0:
        command += ["--max-model-len", str(max_model_len)]
    if max_num_batched_tokens > 0:
        command += ["--max-num-batched-tokens", str(max_num_batched_tokens)]
        
    proc = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT,)

    logger.debug(f"Waiting for the server to be ready...")

    success = False
    while True:
        try:
            r = requests.get(f"{base_url}/v1/models", timeout=2)
            
            if r.status_code == 200:
                logger.debug("Server is ready!")
                success = True
                break
        except requests.RequestException:
            logger.debug("Server is not ready yet...")
            pass  # still not ready    
        
        poll = proc.poll()
        logger.debug(f"process poll: {poll}")
        
        if poll is not None: # Check if the process is still running: None -> still running
            logger.error(f"Serving vLLM crashed. Check logs: {log_file_name}")
    
    log_file.close()
    
    return success


if __name__ == "__main__":
    t_model = "meta-llama/Llama-3.1-8B-Instruct"
    create_test_environment(
        k8s_name=ComponentsYaml.get_k8s_name(model=t_model),
        in_cluster=False,
        verify_ssl=False,
        model=t_model,
        pvc_name="vllm-support",
        image="quay.io/dataprep1/data-prep-kit/vllm_image:0.1",
    )
