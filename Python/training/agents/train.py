import torch
import numpy as np
import os
import time
from datetime import datetime
from collections import deque
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.base_env import ActionTuple
from multiprocessing import Pool
import multiprocessing
import sys

from utils import (
    load_config, find_config_files, AccuracyTracker,
    save_rewards_to_file, save_accuracy_to_file,
    load_rewards_from_csv, load_accuracy_from_csv,
)

ALGORITHMS = {
    'ppo':   lambda: __import__('algorithms.ppo',   fromlist=['PPOAgent']).PPOAgent,
    'coach': lambda: __import__('algorithms.coach', fromlist=['COACHAgent']).COACHAgent,
}

MAX_PARALLEL = 10
WORKER_START_DELAY = 5
VR_MODE = True

def set_global_seeds(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_worker(args):
    config_file, worker_id = args
    delay = worker_id * WORKER_START_DELAY
    if delay > 0:
        time.sleep(delay)
    return train(config_file, worker_id)


def train(config_path="../../configs/config1", worker_id=0):
    #UNITY_BUILD_PATH = "../../../Data/Builds/BaselineHeadless/InteractiveRL.exe"
    UNITY_BUILD_PATH = None

    RESUME_FROM = "training_logs/run_seed42_config3 (2)_fresh_20260529_202921/model_final.pt"
    RESUME_ACCURACY_CSV = "training_logs/run_seed42_config3 (2)_fresh_20260529_202921/accuracy_final.csv"
    RESUME_CSV = "training_logs/run_seed42_config3 (2)_fresh_20260529_202921/rewards_final.csv"
    # RESUME_FROM = None
    # RESUME_ACCURACY_CSV = None
    # RESUME_CSV = None

    cfg = load_config(config_path)
    run_seed = cfg.get('reproducibility', {}).get('seed', 41)
    set_global_seeds(run_seed)

    behavior_cfg = cfg['behaviors']['PepperGreeting']
    hp = behavior_cfg['hyperparameters']
    env_cfg = cfg['env_settings']
    engine_cfg = cfg['engine_settings']

    max_total_steps = behavior_cfg['max_steps']
    ROLLOUT_STEPS = behavior_cfg['rollout_steps']
    ENTROPY_END = hp['entropy_end']

    # Connect to Unity
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(
        time_scale=engine_cfg['time_scale'],
        target_frame_rate=60,
        capture_frame_rate=60,
    )
    env_params_channel = EnvironmentParametersChannel()
    env_params_channel.set_float_parameter("npc_seed", float(run_seed))

    no_graphical_interface = False if VR_MODE else True

    env = UnityEnvironment(
        file_name=UNITY_BUILD_PATH,
        side_channels=[channel, env_params_channel],
        no_graphics=no_graphical_interface,
        worker_id=worker_id,
        timeout_wait=180
    ) 

    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    algo_name = cfg.get('algorithm', 'ppo')

    if VR_MODE:
        algo_name = cfg.get('algorithm', 'coach')

    AgentClass = ALGORITHMS[algo_name]()
    agent = AgentClass(cfg, device=device)

    tracker = AccuracyTracker()

    # Resume
    step_offset = 0
    episode_offset = 0
    prior_rewards = []

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"Loading checkpoint: {RESUME_FROM}")
        loaded_steps, loaded_episodes = agent.load(RESUME_FROM)
        print(f"  Entropy restored: {agent.entropy_coef:.4f}")
        if not RESUME_CSV:
            step_offset = loaded_steps
            episode_offset = loaded_episodes
    elif RESUME_FROM:
        print(f"WARNING: checkpoint not found at '{RESUME_FROM}' - starting fresh.")

    if RESUME_CSV:
        prior_rewards, step_offset, episode_offset = load_rewards_from_csv(RESUME_CSV)

    if RESUME_FROM is None:
        agent.entropy_coef = hp['entropy_start']

    ENTROPY_START = agent.entropy_coef
    action_dist_interval = 10 if VR_MODE else 500

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = "resumed" if RESUME_FROM else "fresh"
    config_name = os.path.splitext(config_path.split("\\")[-1].split("/")[-1])[0]
    output_dir = f"training_logs/run_seed{run_seed}_{config_name}_{run_label}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Redirect stdout to log file
    log_file = open(os.path.join(output_dir, f'{config_name}_logs.log'), 'w')
    sys.stdout = log_file

    with open(os.path.join(output_dir, 'run_info.txt'), 'w') as f:
        f.write(f"Run started      : {timestamp}\n")
        f.write(f"Device           : {device}\n")
        f.write(f"Config file      : {config_path}\n")
        f.write(f"Resumed model    : {RESUME_FROM or 'scratch'}\n")
        f.write(f"Step offset      : {step_offset}\n")
        f.write(f"Episode offset   : {episode_offset}\n")
        f.write(f"Max steps (run)  : {max_total_steps}\n")
        f.write(f"Entropy start    : {ENTROPY_START:.4f}\n")

    print(f"\nTraining on {device} | logs -> {output_dir}")
    print(f"Config: {config_path}")
    print(f"Episodes: {episode_offset} -> ...")
    print(f"Steps: {step_offset} -> {step_offset + max_total_steps}\n")

    all_rewards = list(prior_rewards)
    accuracy_history = load_accuracy_from_csv(RESUME_ACCURACY_CSV) if RESUME_ACCURACY_CSV else []
    episode_rewards_buffer = deque(maxlen=100)
    for _s, r in prior_rewards[-100:]:
        episode_rewards_buffer.append(r)

    total_steps = step_offset
    episode_count = episode_offset
    episode_reward = 0.0
    rollout_step = 0
    next_value = 0.0

    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    if len(decision_steps) == 0:
        print("No agents found after reset!")
        env.close()
        return

    current_obs = decision_steps.obs[0][0]

    try:
        while total_steps < step_offset + max_total_steps:
            while rollout_step < ROLLOUT_STEPS:
                action, value, log_prob = agent.act(current_obs)

                action_tuple = ActionTuple()
                action_tuple.add_discrete(np.array([[action]]))
                env.set_actions(behavior_name, action_tuple)
                env.step()

                decision_steps, terminal_steps = env.get_steps(behavior_name)

                # Determine if episode ended and get reward for THIS action
                if len(terminal_steps) > 0:
                    reward = terminal_steps.reward[0]
                    done = True
                    next_value = 0.0  # terminal state has no future value
                    agent.store_transition(current_obs, action, reward, value, log_prob, done)
                    tracker.record(reward)
                    episode_reward += reward

                    # Store episode results
                    all_rewards.append((total_steps, episode_reward))
                    episode_rewards_buffer.append(episode_reward)
                    avg100 = np.mean(episode_rewards_buffer)

                    ep_acc = tracker.episode_accuracy()
                    ep_correct, ep_wrong = tracker.episode_counts()
                    accuracy_history.append((total_steps, ep_correct, ep_wrong))
                    tracker.reset_episode()

                    acc_display = f"{ep_acc * 100:.1f}%" if ep_acc is not None else "N/A"
                    lt_acc = tracker.lifetime_accuracy()
                    lt_display = f"{lt_acc * 100:.1f}%" if lt_acc is not None else "N/A"

                    print(f"Episode {episode_count:5d} | "
                          f"Steps: {total_steps:7d} | "
                          f"Reward: {episode_reward:7.3f} | "
                          f"Avg(100): {avg100:7.3f} | "
                          f"Entropy: {agent.entropy_coef:.4f} | "
                          f"Acc: {acc_display} ({ep_correct}+ {ep_wrong}-) | "
                          f"Lifetime: {lt_display}")

                    episode_count += 1

                    env.reset()
                    decision_steps, _ = env.get_steps(behavior_name)

                    if len(decision_steps) == 0:
                        raise RuntimeError(
                            f"No agents found after mid-rollout reset at step {total_steps}. "
                            "Cannot continue with a fake zero observation."
                        )
                    current_obs = decision_steps.obs[0][0]
                    episode_reward = 0.0

                else:
                    reward = decision_steps.reward[0]
                    done = False
                    next_obs = decision_steps.obs[0][0]
                    next_value = agent.get_value(next_obs)  # bootstrap value for non-terminal
                    agent.store_transition(current_obs, action, reward, value, log_prob, done)
                    tracker.record(reward)
                    episode_reward += reward
                    current_obs = next_obs

                total_steps += 1
                rollout_step += 1

                # use recent counts only, reset after display
                # vr
                if total_steps % action_dist_interval == 0:
                    total_acts = agent.action_counts_recent.sum()
                    if total_acts > 0:
                        dist_str = " | ".join(
                            f"A{i}:{100 * c / total_acts:.0f}%"
                            for i, c in enumerate(agent.action_counts_recent)
                        )
                        print(f"  [Step {total_steps}] Action dist: {dist_str}")
                    agent.action_counts_recent[:] = 0

                if total_steps >= step_offset + max_total_steps:
                    break

            # Algorithm update
            agent.update(next_value)
            rollout_step = 0

            # Entropy decay
            run_progress = (total_steps - step_offset) / max_total_steps
            agent.entropy_coef = ENTROPY_START + run_progress * (ENTROPY_END - ENTROPY_START)

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    except Exception as e:
        import traceback
        print(f"\nTraining error: {e}")
        traceback.print_exc()
    finally:
        env.close()

        # Save final model
        final_model = os.path.join(output_dir, 'model_final.pt')
        agent.save(final_model, total_steps=total_steps, episode_count=episode_count)
        print(f"Final model -> {final_model}")

        save_rewards_to_file(all_rewards, os.path.join(output_dir, 'rewards_final.csv'), total_steps=total_steps)
        save_accuracy_to_file(accuracy_history, os.path.join(output_dir, 'accuracy_final.csv'))

        # Print final summary
        lt_correct, lt_wrong = tracker.lifetime_counts()
        lt_acc = tracker.lifetime_accuracy()
        if lt_acc is not None:
            bar = '#' * int(lt_acc * 30) + '.' * (30 - int(lt_acc * 30))
            print(f"\n --- Lifetime Accuracy --------------------------")
            print(f"  {bar}  {lt_acc * 100:.1f}%")
            print(f"  Correct: {lt_correct}  |  Wrong: {lt_wrong}  |  "
                  f"Total scored: {lt_correct + lt_wrong}")
            print(f"------------------------------------------")

        print("\n" + "=" * 55)
        print("TRAINING SUMMARY")
        print("=" * 55)
        print(f"Config file      : {config_path}")
        print(f"Steps total      : {total_steps}")
        print(f"Episodes total   : {episode_count}")
        if all_rewards:
            _all_r = [r for _, r in all_rewards]
            print(f"Best reward      : {max(_all_r):.3f}")
            print(f"Avg reward       : {np.mean(_all_r):.3f}")
        print(f"Last 100 avg     : {np.mean(list(episode_rewards_buffer)):.3f}")
        print(f"Output dir       : {output_dir}")
        print("=" * 55)

        sys.stdout = sys.__stdout__
        log_file.flush()
        log_file.close()

        return {
            'label': config_name,
            'rewards': all_rewards,
            'accuracy_history': accuracy_history,
        }


if __name__ == '__main__':
    if VR_MODE:
        config_files = find_config_files('../../configs/Finetune')
        if not config_files:
            print("No config files found in configs/Finetune/ directory!")
            exit(1)

        print("\n" + "=" * 70)
        print(f"VR MODE - config: {config_files[0]}")
        print("=" * 70)
        print("Waiting for Unity Editor - Press play now ... \n")
        print("=" * 70 + "\n")
        train(config_files[0], worker_id = 0)
    
    else:
        multiprocessing.set_start_method('spawn', force=True)
        config_files = find_config_files('../../configs')
        if not config_files:
            print("No config files found in configs/ directory!")
            exit(1)

        print("\n" + "=" * 70)
        print("MULTI-CONFIG TRAINING")
        print("=" * 70)
        print(f"Found {len(config_files)} config files:")
        for i, cf in enumerate(config_files, 1):
            print(f"  {i}. {cf}")
        print(f"Running up to {MAX_PARALLEL} configs in parallel")
        print("=" * 70 + "\n")

        indexed_configs = [(cf, i) for i, cf in enumerate(config_files)]

        with Pool(processes=MAX_PARALLEL) as pool:
            results = pool.map(train_worker, indexed_configs)
            all_run_results = [r for r in results if r is not None]