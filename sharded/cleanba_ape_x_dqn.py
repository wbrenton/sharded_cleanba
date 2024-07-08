import os
import time
import uuid
import queue
import random
import threading
from functools import partial
from collections import deque
from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal, Union, Sequence, Callable, NamedTuple, Any # TODO remove

import tyro
import envpool
from einops import rearrange # TODO use or delete
from rich.pretty import pprint
from clu.metric_writers import SummaryWriter

import jax
import chex
import rlax
import flax
import optax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax.experimental.shard_map import shard_map
from jax.sharding import SingleDeviceSharding, NamedSharding, Mesh, PartitionSpec as PS

# NOTE: for later
    # TODO: move timer to import
    # TODO clu profiling (last)
    # TODO HLO logging

# # Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = "0.6"
os.environ['XLA_FLAGS'] = "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
# # Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
# # os.environ['TF_XLA_FLAGS'] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
# os.environ['TF_CUDNN DETERMINISTIC'] = "1"

# TODO: multi node training
# TODO CARBS tuning on Atari 5
# TODO MuZero 
# and your done, hell of a run 

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    "the name of this experiment"
    seed: int = 1
    "seed of the experiment"
    track: bool = False
    "if toggled, this experiment will be tracked with Weights and Biases"
    wandb_project_name: str = "sebulba2"
    "the wandb's project name"
    wandb_entity: str = None
    "the entity (team) of wandb's project"
    capture_video: bool = False
    "whether to capture videos of the agent performances (check out `videos` folder)"
    save_model: bool = False
    "whether to save model into the `runs/{run_name}` folder"
    upload_model: bool = False
    "whether to upload the saved model to huggingface"
    hf_entity: str = ""
    "the user or org name of the model repository from the Hugging Face Hub"
    log_frequency: int = 1
    "the logging frequency of the model performance (in terms of `updates`)"
    block: bool = True
    "whether or not to block jit compiled function"

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    "the id of the environment"
    total_timesteps: int = 50000000
    "total timesteps of the experiments"
    learning_rate: float = 2.5e-4
    "the learning rate of the optimizer"
    local_batch_size: int = 128
    "the number of examples per gradient update"
    local_num_envs: int = 128
    "the number of parallel game environments"
    actors_per_device: int = 1
    "the number of actor threads to use"
    num_actor_steps: int = 64
    "the number of steps to run in each environment per policy rollout"
    td_steps: int = 3
    "the number of steps to bootstrap the value estimate"
    gamma: float = 0.99
    "the discount factor gamma"
    gradient_accumulation_steps: int = 1
    "the number of gradient accumulation steps before performing an optimization step"
    max_grad_norm: float = 40.0
    "the maximum norm for the gradient clipping"
    learning_starts: int = 50000
    "the number of global steps before learning starts"
    update_target_frequency: int = 1000 # TODO 2500
    "the frequency of updating the target network"

    epsilon_base: float = 0.4
    "the base value of epsilon greedy distribution"
    epsilon_alpha: float = 7.0
    "the alpha value of epsilon greedy distribution"

    replay_ratio: float = 1 / 50
    "the ratio of gradient updates to environment timesteps"
    local_replay_buffer_capacity: int = 100_000
    "the maximum number of timesteps the replay buffer can hold per process"
    alpha: int = 1.0
    "priority sampling exponent in prioritized experience replay"
    beta: int = 1.0
    "importance weighting exponent in prioritized experience replay"

    actor_device_ids: List[int] = field(default_factory=lambda: [0])
    "the device ids that actor workers will use"
    learner_device_ids: List[int] = field(default_factory=lambda: [1, 2])
    "the device ids that learner workers will use"
    distributed: bool = False
    "whether to use `jax.distirbuted`"
    concurrency: bool = False
    "whether to run the actor and learner concurrently"

    # runtime arguments to be filled in
    num_updates: int = 0
    world_size: int = 0
    local_rank: int = 0
    num_envs: int = 0
    batch_size: int = 0
    num_updates: int = 0
    global_learner_devices: Optional[List[str]] = None
    actor_devices: Optional[List[str]] = None
    learner_devices: Optional[List[str]] = None


PRNGKey = jax.Array
Params = flax.core.FrozenDict

class ShardingConfig(NamedTuple):
    agent: NamedSharding
    buffer: NamedSharding
    batches: NamedSharding
    batch: NamedSharding
    replicated: NamedSharding

@flax.struct.dataclass
class AgentState(TrainState):
    target_params: Params

@flax.struct.dataclass
class Timestep:
    obs: jax.Array
    actions: jax.Array
    rewards: jax.Array
    discounts: jax.Array
    q_values: jax.Array
    priority: jax.Array
    env_ids: jax.Array
    truncations: jax.Array
    terminations: jax.Array
    firststeps: jax.Array # first step of an episode

@flax.struct.dataclass
class Batch:
    obs: jax.Array
    actions: jax.Array
    rewards: jax.Array
    discounts: jax.Array
    bootstrap_obs: jax.Array
    importance_weights: jax.Array
    indexer: jax.Array

@flax.struct.dataclass
class ReplayBufferData:
    obs: jax.Array
    actions: jax.Array
    n_step_rewards: jax.Array
    n_step_discounts: jax.Array
    priority: jax.Array

@flax.struct.dataclass
class ReplayBufferState:
    data: Timestep
    position: jax.Array
    
@dataclass
class ReplayBuffer:
    init: Callable
    add: Callable
    sample: Callable
    update: Callable

# following https://github.com/google/dopamine/blob/4552f69af4763053d87ee4ce6d3da59ca3232f3c/dopamine/jax/networks.py#L302
kernel_init = nn.initializers.lecun_normal() #nn.initializers.variance_scaling(scale=1.0 / np.sqrt(3.0), mode="fan_in", distribution="uniform")
Conv = partial(nn.Conv, padding="SAME", kernel_init=kernel_init)
Dense = partial(nn.Dense, kernel_init=kernel_init)

class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jax.Array):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x.astype(jnp.float32) / 255.0

        # convolutional torso
        x = Conv(32, (8, 8), (4, 4))(x)
        x = nn.relu(x)
        x = Conv(64, (4, 4), (2, 2))(x)
        x = nn.relu(x)
        x = Conv(64, (3, 3), (1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))

        # advantage head
        a = Dense(512)(x)
        a = nn.relu(a)
        advantage = Dense(self.action_dim)(a)

        # value head
        v = Dense(512)(x)
        v = nn.relu(v)
        value = Dense(1)(v)

        return value + (advantage - jnp.mean(advantage, axis=1, keepdims=True))

@jax.jit # NOTE: this will always occur in local_device 0
def prepare_data(data): # TODO test asarray
    return jax.tree.map(lambda *xs: jnp.stack(xs), *data)

batch_batch_q_learning = jax.vmap(jax.vmap(rlax.q_learning))

@jax.jit # NOTE: this will always occur in local_device 0
def calculate_priority(data):
    # n-step bootstrapping
    n_step_rewards = jnp.zeros((args.num_actor_steps, args.local_num_envs), dtype=jnp.float32)
    n_step_discounts = jnp.zeros((args.num_actor_steps, args.local_num_envs), dtype=jnp.float32)
    for i in range(args.num_actor_steps):
        r = jnp.zeros((args.local_num_envs), dtype=jnp.float32) # TODO this is not needed just index above
        d = jnp.ones((args.local_num_envs), dtype=jnp.float32)
        for n in reversed(range(args.td_steps)):
            r += d * data.rewards.at[i + n].get()
            d *= data.discounts.at[i + n].get()
        n_step_rewards = jax.lax.dynamic_update_slice_in_dim(n_step_rewards, r[None,], i, axis=0)
        n_step_discounts = jax.lax.dynamic_update_slice_in_dim(n_step_discounts, d[None,], i, axis=0)
    
    # calculate priority
    td_error = batch_batch_q_learning(
        q_tm1=data.q_values.at[:-args.td_steps].get(),
        a_tm1=data.actions.at[:-args.td_steps].get(),
        r_t=n_step_rewards,
        discount_t=n_step_discounts,
        q_t=data.q_values.at[args.td_steps:].get()
    )
    priority = jnp.maximum(jnp.abs(td_error), 1e-4)
    data = jax.tree.map(lambda x: x.at[:-args.td_steps].get(), data)
    return ReplayBufferData(
        obs=data.obs,
        actions=data.actions,
        n_step_rewards=n_step_rewards,
        n_step_discounts=n_step_discounts,
        priority=priority,
    )

# taken from https://github.com/instadeepai/sebulba/blob/main/sebulba/core.py
class Pipeline(threading.Thread):
    """
    The `Pipeline` shards trajectories into `learner_devices`,
    ensuring trajectories are consumed in the right order to avoid being off-policy
    and limit the max number of samples in device memory at one time to avoid OOM issues.
    """

    def __init__(self, max_size: int, sharding: NamedSharding):
        """
        Initializes the pipeline with a maximum size and the devices to shard trajectories across.

        Args:
            max_size: The maximum number of trajectories to keep in the pipeline.
            learner_devices: The devices to shard trajectories across.
        """
        super().__init__(daemon=True)
        self.sharding = sharding
        self.tickets_queue: queue.Queue = queue.Queue()
        self._queue: queue.Queue = queue.Queue(maxsize=max_size)
        self.should_stop = False

    def run(self) -> None:
        """
        This function ensures that trajectories on the queue are consumed in the right order. The
        start_condition and end_condition are used to ensure that only 1 thread is processing an
        item from the queue at one time, ensuring predictable memory usage.
        """
        while not self.should_stop:
            start_condition, end_condition = self.tickets_queue.get()
            with end_condition:
                with start_condition:
                    start_condition.notify()
                end_condition.wait()

    def stop(self) -> None:
        self.should_stop = True

    def put(self, global_step, actor_network_version, storage) -> None:
        """
        Put a trajectory on the queue to be consumed by the learner.
        """
        start_condition, end_condition = (threading.Condition(), threading.Condition())
        with start_condition:
            self.tickets_queue.put((start_condition, end_condition))
            start_condition.wait()  # wait to be allowed to start

        stacked_storage = prepare_data(storage)
        prioritized_storage = calculate_priority(stacked_storage)
        sharded_storage = jax.device_put(prioritized_storage, self.sharding)

        self._queue.put((global_step, actor_network_version, sharded_storage))

        with end_condition:
            end_condition.notify()  # tell we have finish

    def qsize(self) -> int:
        """Returns the number of trajectories in the pipeline."""
        return self._queue.qsize()

    def get(
        self, block: bool = True, timeout: float | None = None
    ):
        """Get a trajectory from the pipeline."""
        while True:
            try:
                payload = self._queue.get(block=True, timeout=1)
                break
            except:
                continue
        return payload


# taken from https://github.com/instadeepai/sebulba/blob/main/sebulba/core.py
class SharedStorage(threading.Thread):
    """
    A `SharedStorage` is a component that allows networks params to be passed from a
    `Learner` component to `Actor` components.
    """

    def __init__(self, init_value, device: jax.Device):
        super().__init__(daemon=True)
        self.version = 0
        self.value = jax.device_put(init_value, device)
        self.device = device
        self.new_value: queue.Queue = queue.Queue()
        self.should_stop = False

    def run(self) -> None:
        """
        This function is responsible for updating the value of the `ParamSource` when a new value
        is available.
        """
        while not self.should_stop:
            try:
                waiting, version = self.new_value.get(block=True, timeout=1)
                self.value = jax.device_put(jax.block_until_ready(waiting), self.device)
                self.version = version
            except queue.Empty:
                continue

    def stop(self) -> None:
        self.should_stop = True

    def update(self, new_params, version) -> None:
        """
        Update the value of the `ParamSource` with a new value.

        Args:
            new_params: The new value to update the `ParamSource` with.
        """
        self.new_value.put((new_params, version))

    def get(self):
        """Get the current value of the `ParamSource`."""
        return self.value, self.version


class Timer:

    def __init__(self, reduce=Literal['mean", "sum']):
        self.reduce = reduce
        if reduce == "mean":
            self.history = deque(maxlen=10)
        elif reduce == "sum":
            self.history = 0
        else: 
            raise ValueError("reduce must be either 'mean' or 'sum'")

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()
        self.interval = self.end_time - self.start_time
        if self.reduce == "mean":
            self.history.append(self.interval)
        elif self.reduce == "sum":
            self.history += self.interval

    def elapsed(self):
        if self.reduce == "mean":
            return np.mean(self.history)
        elif self.reduce == "sum":
            history = self.history
            self.history = 0
            return history

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


ATARI_MAX_FRAMES = int(
    108000 / 4
)  # 108000 is the max number of frames in an Atari game, divided by 4 to account for frame skipping

def make_env(env_id, seed, num_envs):
    def thunk():
        envs = envpool.make(
            env_id,
            env_type="gym",
            num_envs=num_envs,
            episodic_life=False,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 6
            repeat_action_probability=0.25,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 12
            noop_max=1,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 12 (no-op is deprecated in favor of sticky action, right?)
            full_action_space=True,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) Tab. 5
            max_episode_steps=ATARI_MAX_FRAMES,  # Hessel et al. 2018 (Rainbow DQN), Table 3, Max frames per episode
            reward_clip=True,
            seed=seed,
        )
        envs.num_envs = num_envs
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space
        envs.is_vector_env = True
        return envs

    return thunk

def epsilon_greedy_array(num_envs: int) -> np.ndarray:
    num_envs_range = np.arange(num_envs, dtype=jnp.float32)
    exponent = 1.0 + (num_envs_range / (num_envs - 1)) * args.epsilon_alpha
    return args.epsilon_base ** exponent

def actor_fn(
    key: PRNGKey,
    args: Args,
    pipeline: Pipeline,
    shared_storage: SharedStorage,
    writer: SummaryWriter,
    device_thread_id: int,
    actor_device: jax.Device,
):

    epsilons = epsilon_greedy_array(args.local_num_envs)
    greedy_id = np.argmin(epsilons)
    epsilons = jax.device_put(epsilons, actor_device)
    
    @partial(jax.jit, device=actor_device)
    def get_action(
        params: Params,
        obs: jax.Array,
        key: PRNGKey,
    ):
        obs = jnp.array(obs)
        q_values = agent.apply_fn(params, obs)
        greedy_actions = jnp.argmax(q_values, axis=-1)
        key, subkey1, subkey2 = jax.random.split(key, 3)
        random_actions = jax.random.randint(subkey1, (args.local_num_envs,), 0, envs.single_action_space.n)
        determinant = jax.random.uniform(subkey2, (args.local_num_envs,), minval=0.0, maxval=1.0)
        action = jnp.where(determinant < epsilons, random_actions, greedy_actions)
        return obs, action, q_values, key

    timers = {
        "rollout": Timer(reduce="mean"),
        "pipeline_put": Timer(reduce="mean")
    }

    envs = make_env(
        args.env_id,
        args.seed + jax.process_index() + device_thread_id,
        args.local_num_envs
    )()

    episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
    sequential_episodic_returns = np.zeros((args.local_num_envs,), dtype=np.float32)

    storage = []
    global_step = 0
    obs, _ = envs.reset()
    start_time = time.perf_counter()
    for update in range(1, args.num_updates + 2):

        timers['rollout'].start()
        timers.update({
            "get_params": Timer(reduce="sum"),
            "inference": Timer(reduce="sum"),
            "d2h": Timer(reduce="sum"),
            "env_step": Timer(reduce="sum"),
            "storage": Timer(reduce="sum"),
        })

        num_actor_steps_with_bootstrap = args.num_actor_steps + args.td_steps - len(storage)
        for _ in range(num_actor_steps_with_bootstrap):

            with timers['get_params']:
                params, actor_network_version = shared_storage.get()

            with timers['inference']:
                obs, action, q_values, key = block(get_action)(params, obs, key) # block_until_ready()?

            with timers['d2h']:
                cpu_action = jax.device_get(action)

            with timers['env_step']:
                next_obs, rewards, dones, truncated, info = envs.step(cpu_action)
                global_step += args.num_envs

            with timers['storage']:
                env_id = info['env_id']
                storage.append(
                    Timestep(
                        obs=obs,
                        actions=action,
                        rewards=rewards,
                        discounts=(1. - dones.astype(np.float32)) * args.gamma,
                        q_values=q_values,
                        priority=np.zeros_like(rewards),
                        env_ids=env_id,
                        truncations=truncated.astype(np.int32),
                        terminations=info['terminated'],
                        firststeps=(info['elapsed_step'] == 0).astype(np.int32), # TODO delete uneeded
                        # TODO: recast on device
                    )
                )
            obs = next_obs # NOTE: CRITICAL

            sequential_episodic_returns += info["reward"]
            if info['terminated'][greedy_id] or truncated[greedy_id]:
                writer.write_scalars(global_step, {
                    "charts/episodic_return": sequential_episodic_returns[greedy_id],
                    "charts/episodic_length": info['elapsed_step'][greedy_id],
                })
            sequential_episodic_returns *= 1 - (info['terminated'] + truncated)

            episode_returns[env_id] += info['reward']
            returned_episode_returns[env_id] = np.where(
                info['terminated'] + truncated, episode_returns[env_id], returned_episode_returns[env_id]
            )
            episode_returns[env_id] *= (1 - info['terminated']) * (1 - truncated)
            episode_lengths[env_id] += 1
            returned_episode_lengths[env_id] = np.where(
                info['terminated'] + truncated, episode_lengths[env_id], returned_episode_lengths[env_id]
            )
            episode_lengths[env_id] *= (1 - info['terminated']) * (1 - truncated)
        timers['rollout'].stop()

        # send experience to pipeline
        avg_episodic_return = np.mean(returned_episode_returns)
        with timers['pipeline_put']:
            pipeline.put(global_step, actor_network_version, storage)
        storage = storage[-args.td_steps:]

        if update % args.log_frequency == 0:
            if device_thread_id == 0:
                print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return} \n SPS={int(global_step / (time.perf_counter() - start_time))}")
            metrics = {f"stats/{name}_time": timer.elapsed() for name, timer in timers.items()}
            metrics.update({
                "charts/avg_episodic_return": avg_episodic_return,
                "charts/avg_episodic_length": np.mean(returned_episode_lengths),
                "charts/SPS": int(global_step / (time.perf_counter() - start_time)),
                "charts/SPS_actor": int(args.num_envs * args.num_actor_steps / metrics['stats/rollout_time'])
                # TODO: change name to SPS_rollout in the end
            })
            writer.write_scalars(global_step, metrics)

def make_agent(args: Args, key: PRNGKey):
    obs = np.zeros(args.obs_shape, dtype=np.float32)[None,]
    network = QNetwork(args.action_dim)
    network_params = network.init(key, obs)
    agent = AgentState.create(
        apply_fn=network.apply,
        params=network_params,
        target_params=network_params,
        tx=optax.MultiSteps(
            optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(
                    learning_rate=args.learning_rate, eps=1e-5
                ),
            ),
            every_k_schedule=args.gradient_accumulation_steps,
        ),
    )
    print(network.tabulate(key, obs))
    return agent

def make_replay_buffer(args: Args, shardings: ShardingConfig, mesh: Mesh):
    
    length = args.replay_buffer_length
    envs = args.local_num_envs
    insert_length = args.num_actor_steps
    num_samples = args.num_learner_steps
    per_device_envs = envs // len(args.learner_device_ids)
    per_device_batch_size = args.local_batch_size // len(args.learner_device_ids)

    specs = jax.tree.map(lambda x: x.spec, shardings)

    @partial(jax.jit, in_shardings=shardings.replicated, out_shardings=shardings.buffer)
    def init(timestep: Timestep):

        @partial(shard_map, mesh=mesh, in_specs=PS(), out_specs=specs.buffer)
        def _init(timestep: Timestep):
            return ReplayBufferState(
                data=jax.tree.map(lambda x: jnp.zeros((length, per_device_envs) + x.shape, dtype=x.dtype), timestep),
                position=jnp.zeros((), dtype=jnp.int32)
            )
        return _init(timestep)

    @partial(jax.jit, donate_argnames="state", in_shardings=(shardings.buffer, shardings.buffer.data), out_shardings=shardings.buffer)
    def add(state: ReplayBufferState, timesteps: Timestep):
 
        # NOTE: potential scope variable name conflict?
        # @partial(shard_map, mesh=mesh, in_specs=(specs.buffer, specs.buffer.data), out_specs=(specs.buffer))
        def _add(state: ReplayBufferState, timesteps: Timestep):
            data = jax.tree.map(lambda x, y: jax.lax.dynamic_update_slice_in_dim(x, y, state.position, axis=0), state.data, timesteps)
            position = (state.position + insert_length) % length
            return state.replace(
                data=data,
                position=position, # potential communication? can use learning from docs to check if d2d comms is occuring? exand toolkit
            )
        # NOTE: is it worth checking sharding programatically?
        jax.tree.map(lambda x, y: chex.assert_type(x, y.dtype), timesteps, state.data)
        jax.tree.map(lambda x, y: chex.assert_shape(x, ((insert_length, envs) + y.shape[2:])), timesteps, state.data)
        state = _add(state, timesteps)
        return state

    @partial(jax.jit, donate_argnames="state", in_shardings=(shardings.buffer, shardings.replicated), out_shardings=(shardings.buffer, shardings.batches, shardings.replicated))
    @partial(shard_map, mesh=mesh, in_specs=(specs.buffer, specs.replicated), out_specs=(specs.buffer, specs.batches, specs.replicated))
    def sample(state: ReplayBufferState, key: PRNGKey):
        key, subkey = jax.random.split(key)
        sampling_prioritiy = state.data.priority.reshape(-1) ** args.alpha
        sampling_probs = sampling_prioritiy / sampling_prioritiy.sum()
        # TODO manually assert valid probability distribution
        indexer = jax.random.choice(
            key=subkey,
            a=length * per_device_envs,
            shape=(num_samples, per_device_batch_size),
            replace=True, # TODO make false, replace=False causes learning to not happen, resolve my shard mapping each function
            p=sampling_probs,
        )
        # jax.lax.with_sharding_constraint?
        total_indices = jnp.count_nonzero(sampling_prioritiy)
        importance_weights = (total_indices * sampling_probs[indexer]) ** (-args.beta)
        importance_weights = importance_weights / importance_weights.max(-1, keepdims=True) # max along batch dim
        time_index, envs_index = jnp.unravel_index(indexer, shape=(length, per_device_envs))
        batches = Batch(
            obs=state.data.obs.at[time_index, envs_index].get(),
            actions=state.data.actions.at[time_index, envs_index].get(),
            rewards=state.data.n_step_rewards.at[time_index, envs_index].get(),
            discounts=state.data.n_step_discounts.at[time_index, envs_index].get(),
            bootstrap_obs=state.data.obs.at[time_index + args.td_steps, envs_index].get(),
            importance_weights=importance_weights,
            indexer=indexer,
        )
        return state, batches, key

    @partial(jax.jit, donate_argnames="state", in_shardings=(shardings.buffer, shardings.buffer.data, shardings.buffer.data), out_shardings=(shardings.buffer))
    @partial(shard_map, mesh=mesh, in_specs=(specs.buffer, specs.buffer.data, specs.buffer.data), out_specs=(specs.buffer))
    def update(state: ReplayBufferState, indexer: jax.Array, updated_priority: jax.Array):
        time_index, envs_index = jnp.unravel_index(indexer, shape=(length, per_device_envs))
        return state.replace(data=state.data.replace(
            priority=state.data.priority.at[time_index, envs_index].set(updated_priority)
            )
        )

    return ReplayBuffer(
        init=init,
        add=add,
        sample=sample,
        update=update,
    )

if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.distributed:
        jax.distributed.initialize(
            local_device_ids=range(len(args.learner_device_ids) + len(args.actor_device_ids)),
        )
        print(list(range(len(args.learner_device_ids) + len(args.actor_device_ids))))
    args.world_size = jax.process_count()
    args.local_rank = jax.process_index()
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
    actor_devices = [local_devices[d_id] for d_id in args.actor_device_ids]
    global_learner_devices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.world_size)
        for d_id in args.learner_device_ids
    ]; 
    global_learner_devices = sorted(global_learner_devices, key=lambda d: d.id)
    print("global_learner_devices", global_learner_devices)
    args.global_learner_devices = [str(item) for item in global_learner_devices]
    args.actor_devices = [str(item) for item in actor_devices]
    args.learner_devices = [str(item) for item in learner_devices]
    learner_mesh = Mesh(
        devices=global_learner_devices,
        axis_names=("dp",)
    )
    sharding_fn = lambda spec: NamedSharding(learner_mesh, spec)
    shardings = ShardingConfig(
        agent=sharding_fn(PS()),
        buffer=ReplayBufferState(
            data=sharding_fn(PS(None, "dp")),
            position=sharding_fn(PS())
        ),
        batches=sharding_fn(PS(None, "dp")),
        batch=sharding_fn(PS("dp")),
        replicated=sharding_fn(PS())
    )
    def block(x: Any):
        return jax.block_until_ready(x) if args.block else x

    # runtime args
    args.timesteps_per_rollout = int(args.local_num_envs * args.num_actor_steps)
    args.num_learner_steps = int(args.timesteps_per_rollout * args.replay_ratio)
    args.real_replay_ratio = args.num_learner_steps / args.timesteps_per_rollout
    args.num_envs = args.local_num_envs * args.world_size * args.actors_per_device * len(args.actor_device_ids)
    args.batch_size = args.local_batch_size * args.world_size
    args.num_updates = int(args.total_timesteps * args.replay_ratio)
    assert args.local_num_envs % len(args.learner_device_ids) == 0, \
    "local_num_envs must be divisible by len(learner_device_ids)"
    assert args.local_replay_buffer_capacity // args.local_num_envs, \
    "local_replay_buffer_capacity must be divisible by len(local_num_envs)"
    args.replay_buffer_length = args.local_replay_buffer_capacity // args.local_num_envs
    pprint(args)

    # logging
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{uuid.uuid4()}"
    if args.track and args.local_rank == 0:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.write_hparams(vars(args))

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    learner_keys = jax.device_put(key, shardings.replicated)

    # env setup
    envs = make_env(args.env_id, args.seed, args.local_num_envs)()
    args.obs_shape = envs.single_observation_space.shape
    args.action_dim = envs.single_action_space.n

    # agent setup
    agent = make_agent(args, init_key)
    agent = jax.device_put(agent, shardings.agent)

    # NOTE: jax.lax.bitcast_convert_type can be used to cast with out copying

    # replay buffer setup
    example = ReplayBufferData(
        obs=np.zeros(args.obs_shape, dtype=np.uint8),
        actions=np.zeros((), dtype=np.int32),
        n_step_rewards=np.zeros((), dtype=np.float32),
        n_step_discounts=np.zeros((), dtype=np.float32),
        priority=np.zeros((), dtype=np.float32),
    )
    replay_buffer = make_replay_buffer(args, shardings, learner_mesh)
    example = jax.device_put(example, shardings.replicated)
    buffer_state = replay_buffer.init(example)

    batch_q_learning = jax.vmap(rlax.double_q_learning)

    def ape_x_loss(params: Params, target_params: Params, batch: Tuple):

        @partial(shard_map, mesh=learner_mesh, in_specs=PS("dp"), out_specs=(PS(), PS("dp")))
        def spmd_loss(batch):
            q_tm1 = agent.apply_fn(params, batch.obs)
            q_t_value=agent.apply_fn(target_params, batch.bootstrap_obs)
            q_t_selector=agent.apply_fn(params, batch.bootstrap_obs)
            td_error = batch_q_learning(
                q_tm1=q_tm1,
                a_tm1=batch.actions,
                r_t=batch.rewards,
                discount_t=batch.discounts,
                q_t_value=q_t_value,
                q_t_selector=q_t_selector
            )
            losses = rlax.l2_loss(td_error) * batch.importance_weights
            local_loss = losses.mean()
            loss = jax.lax.pmean(local_loss, "dp")
            return loss, (td_error, q_tm1)

        return spmd_loss(batch)

    ape_x_grad_fn = jax.value_and_grad(ape_x_loss, has_aux=True)

    @jax.jit
    def learner_fn(agent: TrainState, batches: Batch):

        # TODO jit here?
        def step(agent, batch):
            batch = jax.lax.with_sharding_constraint(batch, shardings.batch)
            (loss, (td_error, q_values)), grads = ape_x_grad_fn(agent.params, agent.target_params, batch)
            agent = agent.apply_gradients(grads=grads)
            agent = agent.replace(target_params=optax.periodic_update(
                agent.params, agent.target_params, agent.step, args.update_target_frequency,
            ))
            return agent, dict(loss=loss, td_error=td_error, q_values=q_values)

        agent, metrics = jax.lax.scan(step, agent, batches)
        loss = metrics['loss'].mean()
        q_values = metrics['q_values'].mean()
        priority = jnp.maximum(jnp.abs((metrics['td_error'])), 1e-4)

        return agent, priority, dict(loss=loss, q_values=q_values)

    # actors
    shared_storages = []
    dummy_writer = SimpleNamespace()
    dummy_writer.write_scalars = lambda x, y: None # remove need with conditional based on thread id

    pipeline = Pipeline(max_size=5, sharding=shardings.buffer.data)
    pipeline.start()

    for d_idx, d_id in enumerate(args.actor_device_ids):
        shared_storage = SharedStorage(agent.params, device=local_devices[d_id])
        shared_storage.start()
        shared_storages.append(shared_storage)

        for thread_id in range(args.actors_per_device):
            threading.Thread(
                target=actor_fn,
                kwargs=dict(
                    key=key,
                    args=args,
                    pipeline=pipeline,
                    shared_storage=shared_storage,
                    writer=writer if d_idx == 0 and thread_id == 0 else dummy_writer,
                    device_thread_id=d_idx * args.actors_per_device + thread_id,
                    actor_device=local_devices[d_id],
                ),
            ).start()

    timers = {
        "learner": Timer(reduce="mean"),
        "pipeline_get": Timer(reduce="mean"),
        "replay_buffer_add": Timer(reduce="mean"),
        "replay_buffer_sample": Timer(reduce="mean"),
        "learner_fn": Timer(reduce="mean"),
        "replay_buffer_update": Timer(reduce="mean"),
    }

    # TODO delete
    print(f"timesteps_per_rollout: {args.timesteps_per_rollout}")
    print(f"replay_ratio: {args.replay_ratio}")
    print(f"real_replay_ratio: {args.real_replay_ratio}")
    print(f"num_actor_steps: {args.num_actor_steps}")
    print(f"num_learner_steps: {args.num_learner_steps}")

    # learner
    learner_network_version = 0
    while True:

        timers["learner"].start()

        with timers['pipeline_get']:
            (global_step, actor_network_version, timesteps) = pipeline.get()

        with timers['replay_buffer_add']: # TODO: move inside of learner_fn (last)
            buffer_state = block(replay_buffer.add(buffer_state, timesteps))

        if global_step >= args.learning_starts:
            with timers["replay_buffer_sample"]:
                buffer_state, batches, learner_keys = block(replay_buffer.sample(buffer_state, learner_keys))

            with timers['learner_fn']:
                learner_network_version += args.num_learner_steps
                agent, updated_priority, learner_metrics = block(learner_fn(agent, batches))

            with timers['replay_buffer_update']:
                buffer_state = replay_buffer.update(buffer_state, batches.indexer, updated_priority)

        for shared_storage in shared_storages:
            shared_storage.update(agent.params, learner_network_version)

        timers["learner"].stop()

        if learner_network_version % args.log_frequency == 0:
            print(global_step, f"off-policyness: {learner_network_version - actor_network_version}") # TODO: print off-policyness
            metrics = {f"stats/{name}_time": timer.elapsed() for name, timer in timers.items()}
            if learner_network_version > 0:
                metrics.update({f"losses/{k}": v.item() for k, v in learner_metrics.items()})
                metrics.update({
                    "charts/learning_rate": agent.opt_state[2][1].hyperparams['learning_rate'].item(),
                    "charts/pipeline_qsize": pipeline.qsize(),
                    "charts/SPS_learner": int((args.num_actor_steps * args.local_num_envs) / metrics["stats/learner_time"]),
                    "charts/off-policyness": learner_network_version - actor_network_version
                })
            writer.write_scalars(global_step, metrics)

        if learner_network_version >= args.num_updates:
            break

    if args.save_model and args.local_rank == 0:
        if args.distributed:
            jax.distributed.shutdown()
        agent = flax.jax_utils.unreplicate(agent)
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        vars(args),
                        [
                            agent.params['network_params'],
                            agent.params['actor_params'],
                            agent.params['critic_params'],
                        ],
                    ]
                )
            )
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_envpool_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Network, Actor, Critic),
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "PPO",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
                extra_dependencies=['jax", "envpool", "atari'],
            )

    envs.close()
    writer.close()