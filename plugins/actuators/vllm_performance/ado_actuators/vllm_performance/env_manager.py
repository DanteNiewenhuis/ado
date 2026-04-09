# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from abc import abstractmethod
import asyncio
import json
import logging
from enum import Enum
from typing import Annotated

import pydantic
import ray
from ado_actuators.vllm_performance.deployment_management import (
    DeploymentConflictManager,
)
from ado_actuators.vllm_performance.k8s import K8sEnvironmentCreationError
from ado_actuators.vllm_performance.k8s.manage_components import (
    ComponentsManager,
)
from ado_actuators.vllm_performance.k8s.yaml_support.build_components import (
    ComponentsYaml,
)
from kubernetes.client import ApiException
from pydantic import AfterValidator

from orchestrator.utilities.pydantic import validate_rfc_1123

logger = logging.getLogger(__name__)


class EnvironmentState(Enum):
    """
    Environment state
    """

    NONE = "None"
    READY = "ready"


class Environment(pydantic.BaseModel):
    pass

class BareMetalEnvironment(Environment):
    """
    Environment class representing a deployment environment for a model in a bare metal setting.
    """
    state: Annotated[
        EnvironmentState,
        pydantic.Field(description="Current state of the environment (NONE or READY)"),
    ] = EnvironmentState.NONE
    identifier: Annotated[
        str,
        pydantic.Field(
            description="Unique identifier for the environment, used for tracking"
        ),
    ]
    model: Annotated[
        str,
        pydantic.Field(description="LLM model name (e.g., 'meta-llama/Llama-2-7b-hf')"),
    ]
        
    pid: Annotated[
        int | None,
        pydantic.Field(
            default=None,
            description="Process ID of the environment, used for cleanup"
            ),
    ]
    
    def __init__(self, model: str, identifier: str) -> None:
        logger.debug(f"Creating BareMetalEnvironment with model: {model} and configuration: {identifier}")
        
        super().__init__(model=model, identifier=identifier)

class K8SEnvironment(Environment):
    """
    Environment class representing a deployment environment for a model.

    The k8s_name is automatically generated from the model name and validated
    to be RFC 1123 compliant.
    """

    k8s_name: Annotated[
        str,
        AfterValidator(validate_rfc_1123),
        pydantic.Field(
            description="Kubernetes-compliant name for the deployment, automatically generated from the model name and validated to be RFC 1123 compliant"
        ),
    ] = ""
    state: Annotated[
        EnvironmentState,
        pydantic.Field(description="Current state of the environment (NONE or READY)"),
    ] = EnvironmentState.NONE
    configuration: Annotated[
        str,
        pydantic.Field(
            description="Full deployment configuration as a JSON string containing model, image, GPU/CPU settings, and VLLM parameters"
        ),
    ]
    model: Annotated[
        str,
        pydantic.Field(description="LLM model name (e.g., 'meta-llama/Llama-2-7b-hf')"),
    ]

    @pydantic.model_validator(mode="before")
    @classmethod
    def compute_k8s_name(cls, data: dict) -> dict:
        """
        Compute k8s_name from model if not provided.

        :param data: Input data dictionary
        :return: Data dictionary with k8s_name computed
        """
        if (
            isinstance(data, dict)
            and ("k8s_name" not in data or not data["k8s_name"])
            and "model" in data
        ):
            data["k8s_name"] = ComponentsYaml.get_k8s_name(model=data["model"])
        return data


class EnvironmentsQueue:
    def __init__(self) -> None:
        self.environments_queue = []

    async def wait(self) -> None:
        wait_event = asyncio.Event()
        self.environments_queue.append(wait_event)
        await wait_event.wait()

    def signal_next(self) -> None:
        if len(self.environments_queue) > 0:
            event = self.environments_queue.pop(0)
            event.set()

class EnvironmentManager:
    """
    This is a local environment manager used for the baremetal deployment experiment.
    It does not manage real environments but it is used to keep track of the deployments and their state.
    """
    
    @property
    def active_environments(self) -> int:
        return len(self.in_use_environments) + len(self.free_environments)
    
    def environment_usage(self) -> dict:
        return {"max": self.max_concurrent, "in_use": self.active_environments}
    
    @abstractmethod
    def get_environment(self, model: str, definition: str) -> Environment | None:
        raise NotImplementedError
    
    @abstractmethod
    async def wait_for_env(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def wait_deployment_before_starting(self, env: Environment, request_id: str) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def done_creating(self, identifier: str) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def done_using(self, identifier: str, reclaim_on_completion: bool = False) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def cleanup_failed_deployment(self, identifier: str) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def cleanup(self) -> None:
        raise NotImplementedError

@ray.remote
class BareMetalEnvironmentManager(EnvironmentManager):
    """
    This is a local environment manager that manages the vLLM models deployed in a bare metal setting.
    """
    
    def __init__(self) -> None:
        
        self.in_use_environments: dict[str, BareMetalEnvironment] = {}
        self.free_environments: list[BareMetalEnvironment] = []
        self.environments_queue = EnvironmentsQueue()
        
        self.max_concurrent = 1
    
    @property
    def environments(self) -> dict[str, BareMetalEnvironment]:
        return {**self.in_use_environments, **{env.configuration: env for env in self.free_environments}}
        
    def _get_matching_free_environment(self, identifier: str) -> BareMetalEnvironment | None:
        """
        Find an environment matching a deployment configuration
        :param configuration: The deployment configuration to match
        :return: An already existing environment or None
        """
        
        for id, env in enumerate(self.free_environments):
            if env.identifier == identifier:
                del self.free_environments[id]
                return env
        return None

    def _generate_identifier(self, configuration: dict[str, str]) -> str:
        """
        This is the list of entity parameters that define the environment:
            * model name
            * image name
            * number of gpus
            * gpu type
            * number of cpus
            * memory
            * max batch tokens
            * max number of sequences
            * gpu memory utilization
            * data type
            * cpu offload
        Build entity based environment parameters
        :param values: experiment values
        :return: definition
        """
        env_values = {
            "model": configuration.get("model"),
            "n_gpus": configuration.get("n_gpus"),
            "gpu_type": configuration.get("gpu_type"),
            "tensor_parallel_size": configuration.get("tensor_parallel_size"),
            "max_model_len": configuration.get("max_model_len"),
            "max_num_batched_tokens": configuration.get("max_num_batched_tokens"),
            "max_num_seqs": configuration.get("max_num_seqs"),
        }
        
        # logger.debug(f"Built environment values: {env_values}")
        
        return json.dumps(env_values)
    
    def get_environment(self, model: str, configuration: dict[str, str]) -> Environment | None:
        
        identifier: str = self._generate_identifier(configuration=configuration)
        
        logger.debug(f"Requesting environment for configuration {configuration}")
        
        # Check if there is a free environment matching the configuration
        env = self._get_matching_free_environment(identifier)
        
        if env is None:
            if self.active_environments >= self.max_concurrent:
                # can't create more environments now, need clean up
                if len(self.free_environments) == 0:
                    # No room for creating a new environment
                    logger.debug(
                        f"There are already {self.max_concurrent} actively in use, and I can't create a new one"
                    )
                    return None
                
                # Remove one environment to make room for the new one
                env_to_remove = self.free_environments.pop(0)
                self._del_vllm_instance(env_to_remove)
            
            
            try:
                logger.debug(f"Creating new environment for configuration {configuration}")
                # No free environment available, create a new one
                env = BareMetalEnvironment(model=model, identifier=identifier)
            except Exception as e:
                logger.error(f"Error creating environment for configuration {configuration}: {e}")
                return None
            
        else:
            logger.debug(f"Reusing existing environment for configuration {configuration}")
            
        logger.debug(f"Save environment with key: {env.identifier}")
        self.in_use_environments[env.identifier] = env
        
        return env
    
    async def wait_for_env(self) -> None:
        logging.debug("Waiting for an environment to be available")
    
    def wait_deployment_before_starting(self, env: Environment, request_id: str) -> None:
        logging.debug("Waiting for deployment before starting")
    
    def done_creating(self, configuration: str) -> None:
        logging.debug("Done with creating environment")
    
    def done_using(self, configuration: dict[str, str], reclaim_on_completion: bool = False) -> None:
        """Set the environment with the given configuration to be done. 
        If reclaim is true, the environment will not be added to the free environments and it will be considered as completely removed.

        Args:
            configuration (dict[str, str]): The configuration dictionary for the environment.
            reclaim_on_completion (bool, optional): Defaults to False.
        """
        
        logging.debug(f"Done using environment: {configuration}")
        
        identifier = self._generate_identifier(configuration)
        
        env = self.in_use_environments.pop(identifier, None)
        
        logging.debug(f"Environment popped from in-use environments: {env}")
        
        if env is None:
            logging.error(f"Environment with configuration {configuration} not found in in-use environments.")
            return
        
        if not reclaim_on_completion:
            self.free_environments.append(env)

        # Wake up any other deployment waiting in the queue for a
        # free environment.
        self.environments_queue.signal_next()
    
    def cleanup_failed_deployment(self, configuration: dict[str, str]) -> None:
        identifier = self._generate_identifier(configuration=configuration)
        logging.debug(f"Cleaning up failed deployment: {identifier}")
        
        env = self.in_use_environments.pop(identifier, None)
        
        if env is None:
            logging.error(f"Environment with configuration {configuration} not found in in-use environments for cleanup.")
            return
        
        self._del_vllm_instance(env)
        
    def set_pid(self, configuration: dict[str, str], pid: int) -> None:
        identifier = self._generate_identifier(configuration=configuration)
        
        self.environments[identifier].pid = pid
        
    def get_pid(self, configuration: dict[str, str]) -> int | None:
        identifier = self._generate_identifier(configuration=configuration)
        
        logger.debug(f"Getting PID for environment with configuration {configuration}")
        logger.debug(f"Current environments: {self.environments}")
        
        return self.environments[identifier].pid
    
    def _del_vllm_instance(self, env: BareMetalEnvironment) -> None:
        logger.info(f"Cleaning up environment {env}")
        
        pid = env.pid
        
        logger.debug(f"Environment {env.model} has PID {pid}")
        
        if pid is not None:
            try:
                # Try to kill the process
                import psutil

                process = psutil.Process(pid)
                process.terminate()
                process.wait(timeout=30)
                logger.info(f"Environment {env.model} with PID {pid} terminated successfully.")
            except Exception as e:
                logger.error(f"Error terminating environment {env.model} with PID {pid}: {e}")
        else:
            logger.warning(f"Environment {env.model} does not have a PID set. Skipping termination.")
    
    def cleanup(self):
        """ Stop all running vLLM environments

        Raises:
            NotImplementedError: _description_
        """
        logger.info("Cleaning environments")
        
        for env in list(self.in_use_environments.values()) + self.free_environments:
            self._del_vllm_instance(env)

@ray.remote
class K8SEnvironmentManager(EnvironmentManager):
    """
    This is a Ray actor (singleton) managing environments
    """

    def __init__(
        self,
        namespace: str,
        max_concurrent: int,
        in_cluster: bool = True,
        verify_ssl: bool = False,
        pvc_name: str | None = None,
        pvc_template: str | None = None,
    ) -> None:
        """
        Initialize
        :param namespace: deployment namespace
        :param max_concurrent: maximum amount of concurrent environment
        :param in_cluster: flag in cluster
        :param verify_ssl: flag verify SSL
        :param pvc_name: name of the PVC to be created / used
        :param pvc_template: template of the PVC to be created
        """
        self.in_use_environments: dict[str, K8SEnvironment] = {}
        self.free_environments: list[K8SEnvironment] = []
        self.environments_queue = EnvironmentsQueue()
        self.deployment_conflict_manager = DeploymentConflictManager()
        self.namespace = namespace
        self.max_concurrent = max_concurrent
        self.in_cluster = in_cluster
        self.verify_ssl = verify_ssl

        # component manager for cleanup
        self.manager = ComponentsManager(
            namespace=self.namespace,
            in_cluster=self.in_cluster,
            verify_ssl=self.verify_ssl,
            init_pvc=True,
            pvc_name=pvc_name,
            pvc_template=pvc_template,
        )

    def _delete_environment_k8s_resources(self, k8s_name: str) -> None:
        """
        Deletes a deployment. Intended to be used for cleanup or error recovery
        param: identifier: the deployment identifier
        """
        try:
            self.manager.delete_service(k8s_name=k8s_name)
        except ApiException as e:
            if e.reason != "Not Found":
                raise e
        try:
            self.manager.delete_deployment(k8s_name=k8s_name)
        except ApiException as e:
            if e.reason != "Not Found":
                raise e

    async def wait_for_env(self) -> None:
        await self.environments_queue.wait()

    def get_environment(self, model: str, definition: str) -> K8SEnvironment | None:
        """
        Get an environment for definition
        :param model: LLM model name
        :param definition: environment definition - json string containing:
                        model, image, n_gpus, gpu_type, n_cpus, memory, max_batch_tokens,
                        gpu_memory_utilization, dtype, cpu_offload, max_num_seq
        :param increment_usage: increment usage flag
        :return: environment state
        """

        # check if there's an existing free environment satisfying the request
        env = self._get_matching_free_environment(definition)
        if env is None:
            if self.active_environments >= self.max_concurrent:
                # can't create more environments now, need clean up
                if len(self.free_environments) == 0:
                    # No room for creating a new environment
                    logger.debug(
                        f"There are already {self.max_concurrent} actively in use, and I can't create a new one"
                    )
                    return None

                # There are unused environments, let's try to evict one
                environment_evicted = False
                eviction_index = 0
                # Continue looping until we find one environment that can be successfully evicted or we have gone through them all
                while not environment_evicted and eviction_index < len(
                    self.free_environments
                ):
                    environment_to_evict = self.free_environments[eviction_index]
                    try:
                        # _delete_environment_k8s_resources will not raise an error if for whatever the reason the service
                        # or the deployment we are trying to delete does not exist anymore, and we assume
                        # the deployment was properly deleted.
                        self._delete_environment_k8s_resources(
                            k8s_name=environment_to_evict.k8s_name
                        )
                    except ApiException as e:
                        # If we can't delete this environment we try with the next one, but we do not
                        # delete the current env from the free list. This is to avoid spawning more pods than the maximum configured
                        # in the case the failing ones are still running.
                        # Since the current eviction candidate environment will stay in the free ones, some other measurement might
                        # try to evict again and perhaps succeed (e.g., connection restored to the cluster).
                        logger.critical(
                            f"Error deleting deployment or service {environment_to_evict.k8s_name}: {e}"
                        )
                        eviction_index += 1
                        continue

                    logger.info(
                        f"deleted environment {environment_to_evict.k8s_name}. "
                        f"Active environments {self.active_environments}"
                    )
                    environment_evicted = True

                if environment_evicted:
                    # successfully deleted an environment
                    self.free_environments.pop(eviction_index)
                elif len(self.in_use_environments) > 0:
                    # all the free ones have failed deleting but there is one or more in use that
                    # might make room for waiting measurements. In this case we just behave as if there
                    # are no free available environments and we wait.
                    return None
                else:
                    # None of the free environments could be evicted due to errors and none are in use
                    # To avoid a deadlock of the operation we fail the measurement
                    raise K8sEnvironmentCreationError(
                        "All free environments failed deleting and none are currently in use."
                    )

            # We either made space or we had enough space already
            env = K8SEnvironment(model=model, configuration=definition)
            logger.debug(f"New environment created for definition {definition}")

        # If deployments target the same model and the model is not in the HF cache, they would all try to download it.
        # This can lead to corruption of the HF cache data (shared PVC).
        # To avoid this situation, we keep track of the models downloaded by the actuator during the current operation.
        # If a deployment wants to download a model for the first time, we do not allow other deployment using the
        # same model to start in parallel.
        # Once the very first download of a model is done we let any number of deployments using the same model to start
        # in parallel as they would only read the model from the cache.
        self.deployment_conflict_manager.maybe_add_deployment(
            k8s_name=env.k8s_name, model=model
        )

        self.in_use_environments[env.k8s_name] = env

        return env

    def get_experiment_pvc_name(self) -> str:
        return self.manager.pvc_name

    def done_creating(self, identifier: str) -> None:
        """
        Report creation
        :param identifier: environment identifier
        :return: None
        """
        self.in_use_environments[identifier].state = EnvironmentState.READY
        model = self.in_use_environments[identifier].model

        self.deployment_conflict_manager.signal(k8s_name=identifier, model=model)

    def cleanup_failed_deployment(self, identifier: str) -> None:
        env = self.in_use_environments[identifier]
        self._delete_environment_k8s_resources(k8s_name=identifier)
        self.done_using(identifier=identifier, reclaim_on_completion=True)
        self.deployment_conflict_manager.signal(
            k8s_name=identifier, model=env.model, error=True
        )

    def _get_matching_free_environment(self, configuration: str) -> K8SEnvironment | None:
        """
        Find a deployment matching a deployment configuration
        :param configuration: The deployment configuration to match
        :return: An already existing deployment or None
        """
        for id, env in enumerate(self.free_environments):
            if env.configuration == configuration:
                del self.free_environments[id]
                return env
        return None

    async def wait_deployment_before_starting(
        self, env: K8SEnvironment, request_id: str
    ) -> None:
        await self.deployment_conflict_manager.wait(
            request_id=request_id, k8s_name=env.k8s_name, model=env.model
        )

    def done_using(self, identifier: str, reclaim_on_completion: bool = False) -> None:
        """
        Report test completion
        :param definition: environment definition
        :param reclaim_on_completion: flag to indicate the environment is to be completely removed and not freed for later use
        :return: None
        """
        env = self.in_use_environments.pop(identifier)
        if not reclaim_on_completion:
            self.free_environments.append(env)

        # Wake up any other deployment waiting in the queue for a
        # free environment.
        self.environments_queue.signal_next()

    def cleanup(self) -> None:
        """
        Clean up environment
        :return: None
        """
        logger.info("Cleaning environments")
        all_envs = list(self.in_use_environments.values()) + self.free_environments
        for env in all_envs:
            self._delete_environment_k8s_resources(k8s_name=env.k8s_name)

        # We only delete the PVC if it was created by this actuator
        if self.manager.pvc_created:
            logger.debug("Deleting PVC")
            self.manager.delete_pvc()
        else:
            logger.debug("No PVC was created. Nothing to delete!")
