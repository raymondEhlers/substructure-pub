"""Functionality related to job submission

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import os.path
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import attr

from parsl.addresses import address_by_hostname
from parsl.config import Config
from parsl.dataflow.futures import AppFuture
from parsl.executors import HighThroughputExecutor
from parsl.executors.high_throughput.errors import WorkerLost
from parsl.providers import LocalProvider, SlurmProvider
from parsl.launchers import SingleNodeLauncher, SrunLauncher
from parsl.launchers.launchers import Launcher
from parsl.monitoring.monitoring import MonitoringHub

from mammoth import helpers

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

logger = logging.getLogger(__name__)


FACILITIES = Literal[
    "local",
    "test_local_rehlers",
    "ORNL_b587_short",
    "ORNL_b587_long",
    "ORNL_b587_loginOnly",
    "ORNL_b587_vip",
]


def _expand_vars_in_work_dir(
    value: Union[str, Path],
) -> Path:
    """Validate work dir."""
    _p = os.path.expandvars(str(value))
    p = Path(_p)
    return p


@attr.define
class TaskConfig:
    """Configuration for a single task.

    Attributes:
        name: Name of the task. It will be passed to parsl.
        n_cores_per_task: Number of cores required per task.
        memory_per_task: Memory required per task in GB.

    Note:
        We prefer to specify jobs by number of cores (easy to reason in, and it's usually our
        constraint), but some facilities require specifying the memory as well. This is only
        worth doing if absolutely required.
    """

    name: str
    n_cores_per_task: int
    memory_per_task: Optional[int] = attr.field(default=None)


@attr.define
class NodeSpec:
    n_cores: int
    # Denoted in GB.
    memory: int


@attr.define
class Facility:
    """Facility configuration.

    Attributes:
        name: Name of the facility.
        node_spec: Specification for a single node. This is needed to inform parsl
            about node constraints, what resources to request, and how to schedule jobs.
        partition_name: Name of the partition.
        target_allocate_n_cores: Target number of cores to allocate via slurm. This
            may be an entire node, or only part of one. Note that this is separate
            from the number of cores that are required for a particular task. Default: None,
            which corresponds to targeting allocating of the entire node.
        allocation_account: Name of allocation account, to be passed via slurm. Default: "".
        task_configs: Node configurations required for particular tasks. For example,
            for a jet energy loss calculation, or for hydro.
        node_work_dir: Work directory for where jobs are executed. This can be used to execute on
            local storage of a node. Default: Current directory.
        storage_work_dir: Work directory for where runs are stored. Default: Same as the node_work_dir.
        directories_to_mount_in_singularity: Directories to mount in singularity. Default: [].
        worker_init_script: Worker initialization script. Default: "".
        high_throughput_executor_additional_options: Additional keyword options to pass
            directly to the high throughput executor. Default: {}
        launcher: Launcher class to use with the high throughput executor. Default: SrunLauncher.
        parsl_config_additional_options: Additional keyword options to pass directly to
            the parsl config. Default: {}
    """

    name: str
    node_spec: NodeSpec
    partition_name: str
    # Number of cores to target allocating. Default: Full node.
    _target_allocate_n_cores: Optional[int] = attr.field(default=None)
    allocation_account: str = attr.field(default="")
    task_configs: Dict[str, TaskConfig] = attr.Factory(dict)
    node_work_dir: Path = attr.field(default=Path("."))
    storage_work_dir: Path = attr.field(
        converter=_expand_vars_in_work_dir, default=Path(".")
    )
    directories_to_mount_in_singularity: List[Path] = attr.Factory(list)
    worker_init_script: str = attr.field(default="")
    high_throughput_executor_additional_options: Dict[str, Any] = attr.Factory(dict)
    launcher: Callable[[], Launcher] = attr.field(default=SrunLauncher)
    parsl_config_additional_options: Dict[str, Any] = attr.Factory(dict)
    cmd_timeout: int = attr.field(default=10)

    @property
    def target_allocate_n_cores(self) -> int:
        if self._target_allocate_n_cores is None:
            return self.node_spec.n_cores
        return self._target_allocate_n_cores

    @property
    def allocate_full_node(self) -> bool:
        """True if we are allocating full nodes."""
        # If our target for allocating cores is equal to a single node, then we are allocating a full node.
        return self.node_spec.n_cores == self.target_allocate_n_cores


# Define the facility configurations.
_facilities_configs = {
    f"ORNL_b587_{queue}": Facility(
        name="b587",
        node_spec=NodeSpec(n_cores=11, memory=64),
        partition_name=queue,
        # Allocate full node:
        # target_allocate_n_cores=11 if queue != "loginOnly" else 6,
        # Allocate by core:
        target_allocate_n_cores=1,
        launcher=SingleNodeLauncher,
        #node_work_dir=Path("/tmp/parsl/$USER"),
        #storage_work_dir=Path("/alf/data/rehlers/jetscape/work_dir"),
    ) for queue in ["short", "long", "loginOnly", "vip"]
}


def _default_parsl_config_kwargs(workflow_name: str, enable_monitoring: bool = True) -> Dict[str, Any]:
    """Default parsl config keyword arguments.

    These are shared regardless of the facility.

    Args:
        enable_monitoring: If True, enable parsl monitoring. Default: True.

    Returns:
        Default config keyword arguments.
    """
    config_kwargs = dict(
        # This strategy is required to scale down blocks in the HTEX. As of Feb 2021, it is only
        # available in the parsl master.
        strategy="htex_auto_scale",
        # Identify a node as being idle after 20 seconds.
        # This is a balance - if we're too aggressive, then the blocks may be stopped while we still
        # have work remaining. However, if we're not aggressive enough, then we're wasting our allocation.
        max_idletime=20,
    )

    # Setup
    # Monitoring Information
    if enable_monitoring:
        config_kwargs["monitoring"] = MonitoringHub(
            hub_address=address_by_hostname(),
            hub_port=55056,
            monitoring_debug=False,
            resource_monitoring_interval=10,
            workflow_name=workflow_name,
        )

    return config_kwargs


def config(
    facility: FACILITIES,
    task_config: TaskConfig,
    n_tasks: int,
    walltime: str,
    enable_monitoring: bool = False,
    request_n_blocks: Optional[int] = None,
    additional_worker_init_script: str = "",
) -> Tuple[Config, Facility, List[helpers.LogMessage]]:
    """Retrieve the appropriate parsl configuration for a facility and task.

    This is the main interface for retrieving these configurations.

    Args:
        facility: Name of facility. Possible values are in `FACILITIES`.
        task_config: Task configuration to be executed.
        n_tasks: Total number of tasks execute.
        walltime: Requested wall time for the job. Short times will (probably) be easier to schedule.
            Format: "hh:mm:ss".
        enable_monitoring: If True, enable parsl monitoring. Default: False, since I am unsure of how
            this will interact with the particular facilities.
        request_n_blocks: Explicitly request n_blocks instead of the calculated number. This
            value is still validated and won't be blindly accepted. Default: None, which will
            use the calculated number of blocks.
        additional_worker_init_script: Additional script for initializing the worker. Default: ""

    Returns:
        Tuple of: A parsl configuration for the facility - allocating enough blocks to immediately
            execute all tasks, facility config, stored log messages.
    """
    # Validation
    if facility not in _facilities_configs:
        raise ValueError(f"Facility '{facility}' is invalid. Possible values: {_facilities_configs}")
    _facility = _facilities_configs[facility]
    # Create the work directory once we know the facility.
    _facility.storage_work_dir.mkdir(parents=True, exist_ok=True)

    # Further validation
    if "test_local" in facility:
        # We need to treat the case of the local facility differently because
        # the provide is different (ie. it's not slurm).
        return _define_local_config(
            n_tasks=n_tasks,
            task_config=task_config,
            facility=_facility,
            walltime=walltime,
            enable_monitoring=enable_monitoring,
            request_n_blocks=request_n_blocks,
            additional_worker_init_script=additional_worker_init_script,
        )
    else:
        return _define_config(
            n_tasks=n_tasks,
            task_config=task_config,
            facility=_facility,
            walltime=walltime,
            enable_monitoring=enable_monitoring,
            request_n_blocks=request_n_blocks,
            additional_worker_init_script=additional_worker_init_script,
        )


def _define_config(
    n_tasks: int,
    task_config: TaskConfig,
    facility: Facility,
    walltime: str,
    enable_monitoring: bool,
    request_n_blocks: Optional[int] = None,
    additional_worker_init_script: str = "",
) -> Tuple[Config, Facility, List[helpers.LogMessage]]:
    """Define the parsl config based on the facility and task.

    Args:
        n_tasks: Number of tasks to be executed.
        task_config: Task configuration to be executed.
        facility: Facility configuration.
        walltime: Wall time for the job.
        enable_monitoring: If True, enable parsl monitoring. I am unsure of how this will
            interact with the particular facilities.
        request_n_blocks: Explicitly request n_blocks instead of the calculated number. This
            value is still validated and won't be blindly accepted. Default: None, which will
            use the calculated number of blocks.
        additional_worker_init_script: Additional script for initializing the worker. Default: ""

    Returns:
        Tuple of: A parsl configuration for the facility - allocating enough blocks to immediately
            execute all tasks, facility config, stored log messages.
    """
    # Setup
    log_messages: List[helpers.LogMessage] = []

    # Determine request properties.
    # Namely, we need to know:
    # 1. How many cores to request per block
    # 2. How much memory to request per block
    # 3. How many blocks are required to run all tasks.
    n_cores_required = int(n_tasks * task_config.n_cores_per_task)
    if n_cores_required <= facility.target_allocate_n_cores:
        # Only need a single block
        n_blocks = 1
        n_cores_to_allocate_per_block = n_cores_required
        n_tasks_per_block = n_tasks
    else:
        # Need multiple blocks.
        # Let's spread out as evenly as possible.
        # If we tried to pack into as few of blocks as possible, we would waste cores in our allocation.
        n_blocks = math.ceil(n_cores_required / facility.target_allocate_n_cores)
        # Need ceil here in case the number of required cores doesn't divide evenly.
        n_cores_to_allocate_per_block = math.ceil(n_cores_required / n_blocks)
        # Need to make sure that it fits within our core requirements, so round up to the
        # nearest multiple of n_cores_per_task
        n_cores_to_allocate_per_block = n_cores_to_allocate_per_block + (
            n_cores_to_allocate_per_block % task_config.n_cores_per_task
        )
        # Have to additional round because otherwise python will treat this as a float.
        n_tasks_per_block = round(n_cores_to_allocate_per_block / task_config.n_cores_per_task)

        # Cross check
        assert (
            n_tasks_per_block * n_blocks >= n_tasks
        ), f"Too many tasks per block. n_tasks_per_block: {n_tasks_per_block}, n_blocks: {n_blocks}, n_tasks: {n_tasks}"

    # Calculate the memory required per block
    # NOTE: type ignore because mypy apparently can't figure out that this is not None, even though the check is right there...
    memory_to_allocate_per_block = n_tasks_per_block * task_config.memory_per_task if task_config.memory_per_task else None

    log_messages.append(
        helpers.LogMessage(
            __name__,
            "info",
            f"Requesting {n_cores_to_allocate_per_block} cores in {n_blocks} block(s), with {n_tasks_per_block} tasks per block for {n_tasks} total tasks.",
        )
    )
    log_messages.append(
        helpers.LogMessage(
            __name__,
            "debug",
            f"Requesting {n_cores_required} total cores, {memory_to_allocate_per_block * n_tasks_per_block if memory_to_allocate_per_block else 'no constraint on'} GB total memory.",
        )
    )
    # Validate
    if request_n_blocks:
        if request_n_blocks > n_blocks:
            log_messages.append(
                helpers.LogMessage(
                    __name__,
                    "warning",
                    f"Explicitly requested more blocks than needed. We'll ignore this request and take only the minimum. Requested n_blocks: {n_blocks}, required n blocks: {n_blocks}",
                )
            )
        elif request_n_blocks < n_blocks:
            log_messages.append(
                helpers.LogMessage(
                    __name__,
                    "warning",
                    f"Explicitly requested fewer blocks ({request_n_blocks}) than necessary ({n_blocks}) to run everything simultaneously. Tasks will run sequentially in the requested number of blocks.",
                )
            )
            n_blocks = request_n_blocks

    # Setup
    config_kwargs = _default_parsl_config_kwargs(workflow_name=task_config.name, enable_monitoring=enable_monitoring)

    machines_to_exclude: List[str] = [
        #"pc147"
    ]

    config = Config(
        executors=[
            HighThroughputExecutor(
                label=f"Jetscape_{facility.name}_HTEX",
                address=address_by_hostname(),
                cores_per_worker=task_config.n_cores_per_task,
                # cores_per_worker=round(n_cores_per_node / n_workers_per_node),
                # NOTE: We don't want to set the `max_workers` because we want the number of workers to
                #       be determined by the number of cores per worker and the cores per node.
                working_dir=str(facility.node_work_dir),
                provider=SlurmProvider(
                    # This is how many cores and how much memory we'll request per node.
                    cores_per_node=n_cores_to_allocate_per_block,
                    mem_per_node=memory_to_allocate_per_block,
                    # One block is one node.
                    nodes_per_block=1,
                    # We want n_blocks initially because we will have work for everything immediately.
                    # (useless explicitly requested otherwise).
                    min_blocks=0,
                    max_blocks=n_blocks,
                    init_blocks=n_blocks,
                    partition=facility.partition_name,
                    account=facility.allocation_account,
                    # string to prepend to #SBATCH blocks in the submit
                    # Can add additional options directly to scheduler.
                    scheduler_options=f"#SBATCH --exclude={','.join(machines_to_exclude)}" if machines_to_exclude else "",
                    # Command to be run before starting a worker, such as:
                    # 'module load Anaconda; source activate parsl_env'.
                    worker_init=f"{facility.worker_init_script}; {additional_worker_init_script}" if facility.worker_init_script else additional_worker_init_script,
                    launcher=facility.launcher(),
                    walltime=walltime,
                    # If we're allocating full nodes, then we should request exclusivity.
                    exclusive=facility.allocate_full_node,
                    **facility.high_throughput_executor_additional_options,
                ),
            )
        ],
        **config_kwargs,
    )

    return config, facility, log_messages


def _define_local_config(
    n_tasks: int,
    task_config: TaskConfig,
    facility: Facility,
    walltime: str,
    enable_monitoring: bool,
    request_n_blocks: Optional[int] = None,
    additional_worker_init_script: str = "",
) -> Tuple[Config, Facility, List[helpers.LogMessage]]:
    """Local parsl configuration via process pool.

    This allows for testing parsl locally without needing to be on the facilities or having access
    to a test slurm system. Practically, this means that we'll still use the HighThroughputExecutor,
    but it will be provided via local processes. Careful not to overload your system.

    Our execution scheme is as follows:

    - One block is defined as one node.
    - One node is one core.
    - One job (ie worker) is executed per node.

    Args:
        n_tasks: Number of tasks to be executed.
        task_config: Task configuration to be executed.
        facility: Facility configuration.
        walltime: Wall time for the job.
        enable_monitoring: If True, enable parsl monitoring. I am unsure of how this will
            interact with the particular facilities.
        request_n_blocks: Explicitly request n_blocks instead of the calculated number. This
            value is still validated and won't be blindly accepted. Default: None, which will
            use the calculated number of blocks.
        additional_worker_init_script: Additional script for initializing the worker. Default: ""
    Returns:
        Tuple of: A parsl configuration for the facility - allocating enough blocks to immediately
            execute all tasks, facility config, stored log messages.
    """
    # Setup
    log_messages: List[helpers.LogMessage] = []
    n_blocks_exact = (n_tasks * task_config.n_cores_per_task) / facility.target_allocate_n_cores
    n_blocks = math.ceil(n_blocks_exact)
    log_messages.append(
        helpers.LogMessage(
            __name__,
            "info",
            f"Number of blocks required: {n_blocks_exact}, requesting: {n_blocks}. These need to be close, or we will waste resources on some facilities.",
        )
    )
    # NOTE: This ignores the request_n_blocks. For now, it's not worth the effort, since we can easily test on slurm.

    # Setup
    config_kwargs = _default_parsl_config_kwargs(workflow_name=task_config.name, enable_monitoring=enable_monitoring)
    n_cores = facility.target_allocate_n_cores

    local_config = Config(
        executors=[
            HighThroughputExecutor(
                label=f"Jetscape_{facility.name}_HTEX",
                address=address_by_hostname(),
                cores_per_worker=task_config.n_cores_per_task,
                provider=LocalProvider(
                    # One block is one node.
                    nodes_per_block=1,
                    # We want n_blocks initially because we will have work for everything immediately.
                    # (useless explicitly requested otherwise).
                    min_blocks=1,
                    max_blocks=n_cores,
                    # We try to select one core less for the init so we can see some scaling
                    # if we max out everything.
                    # NOTE: We need at least one block, so if set to just one core, n-1 would break.
                    #       Consequently, we require at least one initial block.
                    init_blocks=max(n_cores - 1, 1),
                    worker_init=f"{facility.worker_init_script}; {additional_worker_init_script}",
                    launcher=facility.launcher(),
                    walltime=walltime,
                ),
            )
        ],
        **config_kwargs,
    )

    return local_config, facility, log_messages


def _cancel_future(job: AppFuture) -> None:
    """Cancel the given app future

    Taken from `coffea.processor.executor`

    Args:
        job: AppFuture to try to cancel
    """
    try:
        # NOTE: This is not implemented with parsl AppFutures
        job.cancel()
    except NotImplementedError:
        pass


def provide_results_as_completed(input_futures: Sequence[AppFuture], timeout: Optional[float] = None, running_with_parsl: bool = False) -> Iterable[Any]:
    """Provide results as futures are completed.

    Taken from `coffea.processor.executor`, with small modifications for parsl specific issues
    around cancelling jobs. Without this change, parsl always seems to hang.
    Their docs note that it is essentially the same as `concurrent.futures.as_completed`,
    but it makes sure not to hold references to futures any longer than strictly necessary,
    which is important if the future holds a large result.

    Args:
        input_futures: AppFutures which will eventually contain results
        timeout: Timeout to wait for results be bailing out. Passed directly to
            `concurrent.futures.wait`. Default: None.
        running_with_parsl: If True, don't wait for futures to cancel (since that's not
            implemented in parsl), and just raise the exception. Without this, parsl seems
            to hang. Default: False.

    Returns:
        Iterable containing the results from futures. They are yielded as the futures complete.
    """
    futures = set(input_futures)
    try:
        while futures:
            try:
                done, futures = concurrent.futures.wait(
                    futures,
                    timeout=timeout,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if len(done) == 0:
                    logger.warning(
                        f"No finished jobs after {timeout}s, stopping remaining {len(futures)} jobs early"
                    )
                    break
                while done:
                    try:
                        yield done.pop().result()
                    except concurrent.futures.CancelledError:
                        pass
                    except WorkerLost as e:
                        logger.warning(f"Lost worker: {e}")
                        pass
            except KeyboardInterrupt as e:
                for job in futures:
                    _cancel_future(job)
                running = sum(job.running() for job in futures)
                logger.warning(
                    f"Early stop: cancelled {len(futures) - running} jobs, will wait for {running} running jobs to complete"
                )
                # parsl can't cancel, so we need to break out ourselves
                # It's most convenient to do this by just re-raising the ctrl-c
                if running_with_parsl:
                    raise e
    finally:
        running = sum(job.running() for job in futures)
        if running:
            logger.warning(
                f"Cancelling {running} running jobs (likely due to an exception)"
            )
        while futures:
            _cancel_future(futures.pop())


def merge_results(a: Dict[Any, Any], b: Dict[Any, Any]) -> Dict[Any, Any]:
    """Merge job results together.

    By convention, we merge into the first dict to try to avoid unnecessary copying.

    Although this should generically work for any object which implements `__add__`,
    it's geared towards histograms.

    Note:
        For the first result, it's often convenient to start with a variable containing an
        empty dict as the argument to a. That way, the merged results will be stored in
        a persistent variable.

    Args:
        a: Job result to be merged into.
        b: Result to be merged with.

    Returns:
        Merged histograms
    """
    # Short circuit if nothing to be done
    if not b and a:
        logger.debug("Returning a since b is None")
        return a
    if not a and b:
        logger.debug("Returning b since a is None")
        return b

    # Ensure we don't miss anything in either dict
    all_keys = set(a) | set(b)

    for k in all_keys:
        a_value = a.get(k)
        b_value = b.get(k)
        # Nothing to be done
        if a_value and b_value is None:
            logger.debug(f"b_value is None for {k}. Skipping")
            continue
        # Just take the b value and move on
        if a_value is None and b_value:
            logger.debug(f"a_value is None for {k}. Assigning")
            a[k] = b_value
            continue
        # At this point, both a_value and b_value should be not None
        assert a_value is not None and b_value is not None

        # Recursive on dict
        if isinstance(a_value, dict):
            logger.debug(f"Recursing on dict for {k}")
            a[k] = merge_results(a_value, b_value)
        else:
            # Otherwise, merge
            logger.debug(f"Merging for {k}")
            a[k] = a_value + b_value

    return a
