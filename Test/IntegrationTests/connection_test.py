# test_connection.py
from mlagents_envs.environment import UnityEnvironment
import time

try:
    env = UnityEnvironment(file_name=None)  # If Unity is already running
    env.reset()
    print("✓ Connected to Unity!")

    behavior_name = list(env.behavior_specs.keys())[0]
    print(f"✓ Behavior found: {behavior_name}")

    env.close()
except Exception as e:
    print(f"✗ Connection failed: {e}")