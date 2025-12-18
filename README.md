# Technical Design Document: VR-Based Interactive Reinforcement Learning System

## 1. System Architecture Overview

### 1.1 System Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Unity Client  │◄──►│  Communication   │◄──►│  Python AI      │
│  (Simulation)   │    │    Middleware    │    │   (Training)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

The system follows a distributed client-server architecture with three main components:

**Unity Client (Simulation Environment)**
* Handles all 3D visualization, physics simulation, and VR interaction
* Manages the greeting scenario with human and robot avatars
* Captures multimodal feedback through VR sensors
* Renders the virtual environment and provides visual feedback

**Communication Middleware**
* Acts as a real-time message bus between Unity and Python
* Handles protocol translation, message serialization, and network communication
* Manages connection states and provides fault tolerance
* Ensures synchronized data exchange between simulation and AI components

**Python AI Server (Training System)**
* Contains all reinforcement learning algorithms and neural networks
* Processes human feedback and integrates it into reward signals
* Manages training pipelines and experience replay
* Handles model evaluation and performance tracking

### 1.2 Hardware & software stack
* **VR headset**: HTC Vive Pro Eye
* **Unity**: 6 LTS. Use OpenXR plugin.
  [Unity Documentation](https://docs.unity3d.com/Manual/index.html)
* **Unity ML-Agents (4.0.0 for C# and mlagents 1.1.0 for python)** — use Unity-to-Python API for actions & observations; provides helpers for logging and inference. (ML-Agents 4.x requires Unity >= 6000.0)
  [GitHub](https://github.com/Unity-Technologies/ml-agents/releases)
* **Python**: 3.10.12 (ML-Agents envs use modern Python; recent ML-Agents upgraded PyTorch 2.1.1)
* **PyTorch**: 2.1.1 (to match ML-Agents envs updates) — used for policy nets and human model
* **ROS2**: Alternative communication framework
* **ZeroMQ/gRPC (or ML-Agents)**: Low-latency streaming and Real-time communication
* **Experiment tracking**: TensorBoard + Weights & Biases.

---

## 2. Algorithmic architecture

### 2.1 Overall design
We are using PPO as the main learner, a Deep-TAMER-style reward model to interpret human VR feedback, and COACH-style corrections to make the agent responsive. They are complementary, not alternatives, and together they form a robust IRL system for multimodal VR teaching.
1. **Agent core (PPO)** trains on environment observations and a composite reward signal:
   `r_total = α_env * r_env + α_human * r_human_model(s,a,t)`
   * `r_env` = engineered environment reward (used for baseline)
   * `r_human_model` = predicted scalar from the human-feedback model (see below)
   * α coefficients are tunable and can be scheduled (warmup, annealing).
2. **Human feedback model (Deep TAMER / reward regressor)**

   * A lightweight neural regressor maps time-aligned multimodal feature vectors to scalar feedback. Trained online with teacher labels (controller + implicit signals mapped to “positive/negative” labels).
   * Outputs both mean prediction and confidence (e.g., Gaussian output or small Bayesian head) used to weight feedback.
3. **Immediate corrective module (COACH-inspired, optional)**

   * When the human provides an explicit instantaneous correction (button press, explicit gesture), generate a high-priority update signal used to nudge policy parameters in the short term (policy gradient shaped by the feedback). This is done in addition to the model-based reward (not instead of).
4. **Credit assignment & temporal alignment**

   * Use eligibility traces (TD(λ)) or short fixed temporal windows to assign a human label to preceding state-action sequence. Also use a small RNN or temporal buffer on the human model to capture short time dependencies.

### 2.2 Losses and update rules (concrete)

* PPO policy loss + value loss as baseline: `L_PPO = L_policy + c1 * L_value - c2 * S_entropy`
* Augment policy gradient with *feedback gradient term* when immediate corrective signal f_t is received: add policy gradient update proportional to `w_conf * f_t * ∇_θ log π(a_t|s_t)`.
* Train human reward model using mean squared error (or negative log likelihood if probabilistic output): `L_hr = (r_human_label - hr_model(f_vec))^2 + β_reg * ||θ||^2`.
* Composite reward used for the PPO update: `r_total = r_env + λ_h * hr_model(f_vec)` where λ_h scales with model confidence and a tunable schedule.

### 2.3 Credit assignment / eligibility

* Maintain a short fixed-length ring buffer (e.g., 2–5 seconds, sampled at env step rate) containing (s, a, t, model_features).
* When human signals arrive, retroactively assign the scalar to the buffer entries using a temporal kernel (e.g., exponential decay). This produces soft labels for several recent time steps (better than hard single-step assignment).
* For more rigorous handling, use *eligibility traces* (TD(λ)) on the critic to propagate feedback over preceding states.

### 2.4 Handling delay & noise (practical)

* **Delay compensation:** timestamp all inputs (controller, gaze, audio) using synchronized clocks; use cross-correlation to estimate average human reaction latency and shift labels accordingly.
* **Noise model:** the human model outputs a variance/confidence; use that to weight the influence of `hr_model` on `r_total`. Optionally model teacher as a noisy oracle with a Beta/Bernoulli model for discrete approvals.
* **Smoothing:** apply exponential smoothing to gaze/hand features to reduce jitter before feeding the model.
* **Teacher calibration phase:** at beginning of each session run a 30–60 s calibration where teacher gives a few known approvals/denials to bootstrap the hr_model and calibrate timings.

### 2.5 How They Work Together
Here is the **actual flow**, very simple:

```
VR teacher → Human Reward Model → Reward → PPO → Agent actions
                         ↓
             COACH correction (only when explicit)
```
---



## 3. Detailed Technical Design

### 3.1 Unity Simulation Architecture

#### 3.1.1 Scene Hierarchy
```
GreetingInteractionScene/
├── Environment/
│   ├── SpatialAnchor
│   ├── Lighting
│   ├── Objects
│   └── Boundaries
├── HumanAvatar/
│   ├── VRController (Human)
│   ├── BodyRig
│   ├── AnimationController
│   └── FeedbackDetector
├── RobotAgent/
│   ├── RobotModel
│   ├── AnimationController
│   ├── StateSensor
│   └── ActionExecutor
├── UI/
│   ├── HUDCanvas
│   ├── FeedbackUI
│   └── TrainingMetrics
└── Managers/
    ├── GameManager
    ├── CommunicationManager
    ├── DataLogger
    └── ConfigManager
```

#### 3.1.2 Core System Components

**Communication Layer**
Provides abstracted interfaces for different communication protocols, allowing seamless data exchange with the Python AI system. Handles message serialization, connection management, and error recovery while maintaining real-time performance.

**Simulation Core**
Manages the fundamental agent-environment interaction loop. Includes state sensing modules that capture environmental information, action execution systems that translate AI decisions into animations, and behavior controllers that manage the robot's responses.

**VR Feedback System**
Processes multimodal input from VR devices including controller buttons, head tracking, gaze analysis, and voice commands. Converts raw sensor data into structured feedback signals with confidence scoring and modality fusion.

### 3.2 Python AI Training Architecture

#### 3.2.1 Core Training Pipeline
Implements the complete reinforcement learning workflow with human feedback integration. Manages episode execution, experience collection, and model updates while coordinating with the Unity simulation.

- Training Loop
- Agent-Environment Interaction


#### 3.2.2 Reinforcement Learning Agent
Implements various RL algorithms with specific adaptations for human feedback integration. Maintains policy networks, value estimators, and exploration strategies while processing multimodal input.

- Policy Architecture
- Feedback Integration


#### 3.2.3 Communication Bridge: Using ML-Agents for Python-Based Training
For this project, we will use **Unity ML-Agents** as a bridge to connect the Unity environment with a Python-based PyTorch implementation of reinforcement learning. This allows all AI logic, including policy networks, training, and optimization, to be handled entirely in Python, without writing any C# code for intelligence inside Unity.

Advantages of this approach:

* **Full Python control**: All neural networks, PPO/SAC algorithms, and training loops are implemented in PyTorch, leveraging the flexibility of Python.
* **Seamless Unity integration**: ML-Agents handles communication between Unity and Python, providing observations, rewards, and done flags directly to the Python trainer.
* **Parallelisation support**: Multiple instances of the Unity environment can run concurrently, letting the agent collect more experience per training step and improving sample efficiency.

The training loop works as follows: the Python agent receives observations from Unity, computes actions with its policy network, sends the actions back to Unity, and receives rewards and episode termination information. This cycle repeats continuously, enabling the agent to learn from interactions entirely through Python while Unity provides the simulation environment.
#### Step loop (what actually happens)
Per Unity physics step:

```
CollectObservations()
↓
Python receives observations
↓
Python sends action
↓
OnActionReceived()
↓
AddReward()
↓
done? EndEpisode()
```
---

## 4. Project Folder Structure Organization

### 4.1 Unity Project Structure
Organizes assets, scripts, and resources following Unity best practices while maintaining clear separation of concerns.

```
VR_IRL_Project/
├── Unity/
│   ├── Assets/
│   │   ├── Scripts/
│   │   │   ├── Communication/
│   │   │   │   ├── ICommunicationInterface.cs
│   │   │   │   ├── ROS2Communicator.cs
│   │   │   │   └── MessageTypes.cs
│   │   │   ├── Agents/
│   │   │   │   ├── RobotAgent.cs
│   │   │   │   ├── HumanAgent.cs
│   │   │   │   └── AgentManager.cs
│   │   │   ├── Sensors/
│   │   │   │   ├── StateSensor.cs
│   │   │   │   ├── DistanceSensor.cs
│   │   │   │   ├── GestureSensor.cs
│   │   │   │   └── StateHistory.cs
│   │   │   ├── Actions/
│   │   │   │   ├── ActionExecutor.cs
│   │   │   │   ├── AnimationController.cs
│   │   │   │   └── ActionTypes.cs
│   │   │   ├── VR/
│   │   │   │   ├── VRFeedbackController.cs
│   │   │   │   ├── HeadPoseTracker.cs
│   │   │   │   ├── GazeTracker.cs
│   │   │   │   ├── VoiceRecognizer.cs
│   │   │   │   └── FeedbackConfig.cs
│   │   │   ├── Environment/
│   │   │   │   ├── GameManager.cs
│   │   │   │   └── TrainingController.cs
│   │   │   ├── UI/
│   │   │   │   ├── HUDManager.cs
│   │   │   │   ├── FeedbackUI.cs
│   │   │   │   └── MetricsDisplay.cs
│   │   │   └── Utilities/
│   │   │       ├── DataLogger.cs
│   │   │       ├── ConfigManager.cs
│   │   │       └── ExtensionMethods.cs
│   │   ├── Scenes/
│   │   │   ├── GreetingInteraction.unity
│   │   │   ├── TrainingEnvironment.unity
│   │   │   └── VRSetup.unity
│   │   ├── Prefabs/
│   │   │   ├── Robot/
│   │   │   ├── HumanAvatar/
│   │   │   └── Environment/
│   │   ├── Animations/
│   │   │   ├── Robot/
│   │   │   │   ├── Wave.anim
│   │   │   │   ├── Handshake.anim
│   │   │   │   └── Bow.anim
│   │   │   └── Human/
│   │   ├── Materials/
│   │   ├── Models/
│   │   └── Resources/
│   ├── Packages/
│   ├── ProjectSettings/
│   └── README.md
├── Python/
│   ├── training/
│   │   ├── agents/
│   │   │   ├── base_agent.py
│   │   │   ├── vr_irl_agent.py
│   │   │   └── feedback_processor.py
│   │   ├── algorithms/
│   │   │   ├── policy_gradient.py
│   │   │   └── multimodal_fusion.py
│   │   ├── core/
│   │   │   ├── training_pipeline.py
│   │   │   └── training_logger.py
│   │   └── models/
│   │       ├── neural_networks.py
│   │       └── policy_networks.py
│   ├── communication/
│   │   ├── unity_bridge.py
│   │   ├── message_protocols.py
│   │   └── ros2_interface.py
│   ├── environments/
│   │   └── greeting_environment.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── comparative_analysis.py
│   │   └── visualization.py
│   ├── utils/
│   │   ├── config_loader.py
│   │   ├── data_management.py
│   │   └── helper_functions.py
│   ├── configs/
│   │   ├── training_config.yaml
│   │   ├── agent_config.yaml
│   │   ├── environment_config.yaml
│   │   └── communication_config.yaml
│   └── requirements.txt
├── Data/
│   ├── training_logs/
│   │   ├── autonomous_rl/
│   │   ├── keyboard_irl/
│   │   └── vr_irl/
│   ├── models/
│   │   ├── checkpoints/
│   │   └── final_models/
│   ├── user_studies/
│   │   ├── qualitative_feedback/
│   │   └── performance_metrics/
│   └── processed/
│       ├── feedback_data/
│       └── state_action_pairs/
├── Documentation/
│   ├── technical_specification.md
│   ├── api_reference.md
│   ├── setup_guide.md
│   ├── user_manual.md
│   └── experiment_protocol.md
├── ThirdParty/
│   ├── PythonROS2Bridge/
│   ├── PyTorch/
│   └── UnityPackages/
├── Tests/
│   ├── unity_tests/
│   ├── python_tests/
│   └── integration_tests/
└── README.md
```

---

## 5. Unity / VR integration & data pipeline

### 5.1 Unity environment

* Use **Unity engine** for the greeting scene (proposal choice). Use **Unity ML-Agents** for instrumentation and communication between Unity and Python training process. ML-Agents is maintained and supports sending observations and receiving actions via a Python API.
* Use **OpenXR** + vendor SDKs to access eye-tracking and hand tracking. The Vive docs and OpenXR eye examples show data access patterns for gaze data in Unity. [Docs](https://developer.vive.com/resources/openxr/unity/)

### 5.2 Communication pattern

Two valid patterns; both are supported by ML-Agents:

A. **ML-Agents default socket/gRPC pipeline**

* Unity runs the environment, streams observations and extra telemetry (multimodal features) to the Python trainer via ML-Agents gRPC socket. Python computes actions and returns them. Use this for synchronous training loop.

B. **Custom sidechannel / websocket** (recommended for richer, lower-latency human channels)

* Use Unity→Python **side-channel** or a dedicated **ZeroMQ / gRPC** link for streaming high-rate sensor data (eye gaze 120Hz, hand poses). Use JSON for messages. The training loop still uses ML-Agents actions; sidechannel provides continuous human signal stream for the human modelling service.


### 5.3 Time sync & timestamps

* Unity timestamps every sensor sample with a monotonic Unity clock. Python side uses the timestamp to align human events with agent timesteps. Record both device timestamps and system timestamps to correct drift.

### 5.4 Data formats

* **Sensor frame:** `{timestamp, head_pose, left_hand_pose, right_hand_pose, gaze_origin, gaze_dir, audio_chunk_id, controller_buttons}`
* **Teacher label message:** `{timestamp, label_type: explicit/implicit, value: +1/-1/0, confidence_source: controller/gaze/nod/ASR, raw_modalities_snapshot_id}`

---



## 6. Data Structures and Message Protocols

### 6.1 Unity to Python Communication

**Agent State Representation**
Encapsulates the complete observable state of the environment including gesture information, action history, etc. 

**Human Feedback Structure**
Standardizes multimodal input from VR devices into a unified format with confidence scoring and modality tracking. Supports both explicit intentional feedback and implicit behavioral cues.

**Feedback Types**
- Explicit ratings through controller input
- Implicit signals from body language and gaze

### 6.2 Python to Unity Communication

**Action Response Format**
Communicates AI decisions back to the simulation with supporting information for analysis and debugging.

**Response Components**
- Action selection and behavioral commands

---

## 7. Configuration Management System

### 7.1 Unity Configuration
Centralized management of simulation parameters, communication settings, and VR configuration with the capability of adjustment in runtime.

```csharp
// Scripts/Utilities/ConfigManager.cs
public class ConfigManager : MonoBehaviour
{
    [Header("Communication Settings")]
    public string pythonHost = "localhost";
    public int feedbackPort = 5558;
    
    [Header("Training Settings")]
    public float timeScale = 1.0f;
    public int maxStepsPerEpisode = 100;
    
    [Header("VR Feedback Settings")]
    public float headNodThreshold = 0.8f;
    public float headShakeThreshold = 0.8f;
    public float gazeAttentionThreshold = 0.6f;
    public float feedbackConfidenceThreshold = 0.7f;
    
    [Header("Robot Settings")]
    public float animationTransitionTime = 0.3f;
    public float actionExecutionTime = 2.0f;
}
```

### 7.2 Python Configuration
Hierarchical configuration system using YAML files for experiment management, algorithm tuning, and system parameters.

```yaml
# configs/training_config.yaml
training:
  max_episodes: 1000
  max_steps_per_episode: 100
  learning_rate: 0.001
  batch_size: 32

agent:
  type: "DQN"  #placeholder
  state_dim: 10
  action_dim: 4
  hidden_layers: [128, 128]

feedback:
  explicit_weight: 1.0
  implicit_weight: 0.7
  confidence_threshold: 0.6
  modality_weights:
    controller: 1.0
    head_pose: 0.8
    gaze: 0.6
    voice: 0.9
  fusion_method: "weighted_average"

environment:
  reward_correct: 10.0
  reward_incorrect: -5.0
  reward_delay: -1.0
  human_feedback_reward_scale: 2.0

communication:
  protocol: "ros2"
  feedback_port: 5558
  timeout_ms: 1000
  retry_attempts: 3
```
---

## 8. Implementation Specifications

This section describes how the previously defined system design is realized in practice. It focuses on implementation responsibilities and execution flow, without repeating architectural, algorithmic, or data structure details already presented in earlier sections.

### 8.1 Implementation Scope

The implementation follows the system design exactly as specified in this document. Unity is used only for simulation, VR interaction, and data collection, while all learning, optimization, and feedback interpretation logic is implemented in Python. No learning logic is embedded inside the Unity project.

The implemented system supports three training modes:

* Autonomous reinforcement learning
* Interactive reinforcement learning with keyboard input
* Interactive reinforcement learning with VR-based multimodal feedback

These modes share the same environment and learning pipeline and differ only in how feedback is generated and weighted.

---

### 8.2 Unity Implementation Responsibilities

On the Unity side, the implementation includes:

* The greeting interaction environment and episode logic
* The robot avatar and its executable greeting actions
* VR device integration and sensor data capture
* Packaging of observations and human feedback into structured messages

Unity acts as a real-time simulator and input collector. It does not evaluate behavior quality, assign meaning to feedback, or update learning models. All collected data is forwarded unchanged to the Python side for processing and learning.

---

### 8.3 Communication Implementation

The communication layer is implemented using **Unity ML-Agents**, which handles the exchange of observations, actions, rewards, and episode signals between Unity and the Python training process. Human feedback that does not follow the agent step loop is transmitted through **ML-Agents side channels**, allowing asynchronous feedback with preserved timestamps. This enables the learning system to correctly align delayed or high-frequency human feedback with agent behavior.

---

### 8.4 Learning System Implementation

The Python implementation integrates the reinforcement learning agent, the human feedback model, and the training loop into a single pipeline. The agent is trained only in Python, using observations and feedback received from Unity.

Human feedback is processed through a learned model before influencing the agent’s updates. This design choice allows the system to handle noisy, delayed, or partial feedback and avoids relying on raw human input as a direct reward signal.

All training runs, intermediate models, and evaluation data are logged automatically to support later comparison and analysis.

---

### 8.5 Temporal Handling and Feedback Assignment

The implementation includes a lightweight mechanism for associating human feedback with recent agent behavior. Instead of assuming perfect timing, feedback is linked to short windows of past interaction.

This allows the system to remain robust to natural human reaction delays and variation in feedback timing, without requiring strict synchronization.

---

### 8.6 Configuration and Reproducibility

All implementation parameters are controlled through configuration files. This includes training settings, feedback weighting, and runtime options.

As a result, different experimental conditions can be reproduced and compared without modifying the implementation code, supporting systematic evaluation.

