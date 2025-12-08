# Technical Design Document: VR-Based Interactive Reinforcement Learning System

## 1. System Architecture Overview

### 1.1 High-Level Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Unity Client  │◄──►│  Communication   │◄──►│  Python AI      │
│  (Simulation)   │    │    Middleware    │    │   (Training)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

The system follows a distributed client-server architecture with three main components:

**Unity Client (Simulation Environment)**
- Handles all 3D visualization, physics simulation, and VR interaction
- Manages the greeting scenario with human and robot avatars
- Captures multimodal feedback through VR sensors
- Renders the virtual environment and provides visual feedback

**Communication Middleware**
- Acts as a real-time message bus between Unity and Python
- Handles protocol translation, message serialization, and network communication
- Manages connection states and provides fault tolerance
- Ensures synchronized data exchange between simulation and AI components

**Python AI Server (Training System)**
- Contains all reinforcement learning algorithms and neural networks
- Processes human feedback and integrates it into reward signals
- Manages training pipelines and experience replay
- Handles model evaluation and performance tracking

### 1.2 Technology Stack
- **Unity 6**: Simulation environment, VR interface, rendering
- **Python 3.9+**: AI/ML training pipeline
- **PyTorch**: Deep reinforcement learning algorithms
- **Unity ML-Agents**: Environment interface (communication only)
- **ROS2**: Alternative communication framework
- **ZeroMQ/gRPC (or)**: Real-time communication


## 2. Detailed Technical Design

### 2.1 Unity Simulation Architecture

#### 2.1.1 Scene Hierarchy
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

#### 2.1.2 Core System Components

**Communication Layer**
Provides abstracted interfaces for different communication protocols, allowing seamless data exchange with the Python AI system. Handles message serialization, connection management, and error recovery while maintaining real-time performance.

**Simulation Core**
Manages the fundamental agent-environment interaction loop. Includes state sensing modules that capture environmental information, action execution systems that translate AI decisions into animations, and behavior controllers that manage the robot's responses.

**VR Feedback System**
Processes multimodal input from VR devices including controller buttons, head tracking, gaze analysis, and voice commands. Converts raw sensor data into structured feedback signals with confidence scoring and modality fusion.

### 2.2 Python AI Training Architecture

#### 2.2.1 Core Training Pipeline
Implements the complete reinforcement learning workflow with human feedback integration. Manages episode execution, experience collection, and model updates while coordinating with the Unity simulation.

- Training Loop
- Agent-Environment Interaction


#### 2.2.2 Reinforcement Learning Agent
Implements various RL algorithms with specific adaptations for human feedback integration. Maintains policy networks, value estimators, and exploration strategies while processing multimodal input.

- Policy Architecture
- Feedback Integration


#### 2.2.3 Communication Bridge
Establishes reliable bidirectional communication with Unity using publish-subscribe patterns. Handles message routing, data serialization, and connection health monitoring.


## 3. Project Folder Structure Organization

### 3.1 Unity Project Structure
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

## 4. Data Structures and Message Protocols

### 4.1 Unity to Python Communication

**Agent State Representation**
Encapsulates the complete observable state of the environment including gesture information, action history, etc. 

**Human Feedback Structure**
Standardizes multimodal input from VR devices into a unified format with confidence scoring and modality tracking. Supports both explicit intentional feedback and implicit behavioral cues.

**Feedback Types**
- Explicit ratings through controller input
- Implicit signals from body language and gaze

### 4.2 Python to Unity Communication

**Action Response Format**
Communicates AI decisions back to the simulation with supporting information for analysis and debugging.

**Response Components**
- Action selection and behavioral commands


## 5. Configuration Management System

### 5.1 Unity Configuration
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

### 5.2 Python Configuration
Hierarchical configuration system using YAML files for experiment management, algorithm tuning, and system parameters.

### 5.2 Python Configuration
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

## 6. Implementation Specifications

## 7. Evaluation Framework


This technical design provides a comprehensive blueprint for implementing the VR-based interactive reinforcement learning system while maintaining clear separation between simulation (Unity) and AI (Python) components. The architecture supports real-time human-in-the-loop training with multimodal feedback integration and  the system is built to accurately test its performance.