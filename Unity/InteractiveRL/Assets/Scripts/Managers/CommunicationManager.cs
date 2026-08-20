using System.Collections;
using Agents;
using Agents.Human;
using Agents.NPC;
using Agents.Robot;
using Managers.Reward;
using Tasks;
using UnityEngine;

namespace Managers
{
    /// <summary>
    /// Central coordinator for a training episode.
    ///
    /// Responsibilities:
    ///   1. Find Pepper and the human agent (NPC or VR player) at startup.
    ///   2. Forward the ML agent's chosen action to PepperController.
    ///   3. Evaluate and assign rewards based on experiment mode:
    ///      - Mode 0: Autonomous (robot gets rewards based on task success)
    ///      - Mode 1: Keyboard IRL (human presses arrow keys)
    ///      - Mode 2: VR Multimodal (buttons + voice + head gestures)
    ///   4. Reset the scene at the start of every episode.
    ///   5. Expose body-signal observations for the RL policy.
    /// </summary>
    public class CommunicationManager : MonoBehaviour
    {
        public enum ExperimentMode
        {
            Baseline = 0,
            Keyboard = 1,
            VR = 2
        }

        // Inspector
        [Header("Experiment Mode")] [Tooltip("0 = Autonomous Baseline, 1 = Keyboard IRL, 2 = VR Multimodal IRL")]
        public ExperimentMode experimentMode = ExperimentMode.Baseline;

        [Header("Agent References")] [SerializeField]
        private PepperController pepperController;

        [SerializeField] private PepperAgent pepperAgent;

        [Header("Interaction Settings")] [SerializeField]
        private float interactionDistanceThreshold = 3f;

        [SerializeField] private float handshakeCooldown = 3f;

        [Header("Reward Settings")] [Tooltip("Used for debugging for non VR")] [SerializeField]
        private float keyboardRewardAmount = 1f;

        private float currentFeedback = 0.0f;

        [Header("Body Tracking")]
        [Tooltip("Wrist bone of the human agent (NPC rig or XR right hand anchor).")]
        [SerializeField]
        private Transform wristBone;

        [Tooltip("Head bone (NPC rig) or Main Camera (XR). Used to derive person height and gaze direction.")]
        [SerializeField]
        private Transform headOrGazeBone;

        [Tooltip("World-space Y position of the floor. Usually 0.")] [SerializeField]
        private float floorHeight = 0f;

        [Header("Reproducibility")] [Tooltip("Master seed for this run - set from config file")] [SerializeField]
        private int masterSeed = -1;

        private int currentEpisodeNumber = 0;

        [Header("Debug")] [SerializeField] private bool logInteractions = true;

        // Runtime state
        private IHumanAgent humanAgent; // NPC or VR player - same interface

        public IHumanAgent HumanAgent
        {
            get => humanAgent;
            set => humanAgent = value;
        }

        private bool isVRMode;
        private bool isHandshakeInProgress;
        private bool canHandshake = true;
        private int lastRewardedTaskId = -1;
        private NPCController.NPCState lastRewardedNPCState = NPCController.NPCState.Idle;
        private bool taskRewardedThisEvent;

        // Saved transforms for episode reset
        private Vector3 initialPepperPosition;
        private Quaternion initialPepperRotation;
        private Vector3 initialNPCPosition;
        private Quaternion initialNPCRotation;

        private int totalTasksPerformed;

        // Reward Provider
        private IRewardProvider rewardProvider;
        private bool hasActedForCurrentTask = false;

        // Public accessors (read by PepperAgent for observations)
        public PepperController.PepperState CurrentPepperState =>
            pepperController != null ? pepperController.CurrentState : PepperController.PepperState.Idle;

        public NPCController.NPCState CurrentNPCState =>
            humanAgent?.CurrentState ?? NPCController.NPCState.Idle;

        public int CurrentNPCTaskId =>
            humanAgent?.CurrentTask?.id ?? 0;

        public float CurrentDistance => DistanceBetweenAgents();
        public float CurrentFeedback => currentFeedback;

        public bool IsHandshakeInProgress => isHandshakeInProgress;
        public bool CanHandshake => canHandshake;
        public bool IsVRMode => isVRMode;

        public PepperAgent PepperAgent => pepperAgent;
        public PepperController PepperController => pepperController;
        public Transform HumanTransform => humanAgent?.Transform;
        public int GetCurrentEpisodeNumber() => currentEpisodeNumber;

        // Body-signal observations (consumed by PepperAgent.CollectObservations)

        // Wrist height above the floor, normalised to [0, 1].
        // Falls back to 0 when the wrist bone is not assigned.

        private float GetPersonHeight()
        {
            if (headOrGazeBone == null) return 1.8f; // safe fallback if bone not assigned
            return Mathf.Max(0.1f, headOrGazeBone.position.y - floorHeight);
        }

        // Returns the approximate hip/core position derived from head height.
        // Hip sits at ~55 % of standing height for an average adult.
        private Vector3 GetApproximateHipPosition()
        {
            if (headOrGazeBone == null) return Vector3.zero;
            float personHeight = GetPersonHeight();
            return new Vector3(
                headOrGazeBone.position.x,
                floorHeight + personHeight * 0.55f,
                headOrGazeBone.position.z
            );
        }

        // Wrist height as a fraction of the person's own height.
        public float WristHeight
        {
            get
            {
                if (wristBone == null) return 0f;
                float noise = Random.Range(-observationValueNoise, observationValueNoise);
                return (wristBone.position.y - floorHeight) / GetPersonHeight() + noise;

            }
        }

        // Wrist-to-hip distance as a fraction of the person's own height.
        public float WristToCoreDistance
        {
            get
            {
                if (wristBone == null) return 0f;
                float noise = Random.Range(-observationValueNoise, observationValueNoise);
                return Vector3.Distance(wristBone.position, GetApproximateHipPosition()) / GetPersonHeight() + noise;
               
            }
        }

        // Body forward direction relative to Pepper, expressed as a signed angle
        // in [-1, 1] (normalised from [-180 deg, 180 deg]).
        // Falls back to 0 when references are unavailable.
        public float BodyOrientation
        {
            get
            {
                if (humanAgent?.Transform == null || pepperController == null) return 0f;
                Vector3 humanForward = humanAgent.Transform.forward;
                Vector3 toPepper = (pepperController.transform.position - humanAgent.Transform.position).normalized;
                float angle = Vector3.SignedAngle(humanForward, toPepper, Vector3.up);
                return Mathf.Clamp(angle / 180f, -1f, 1f);
            }
        }

        // Gaze (head) direction relative to Pepper, expressed as a signed angle
        // in [-1, 1] (normalised from [-180 deg, 180 deg]).
        // Positive = human is looking towards Pepper.
        // Falls back to BodyOrientation when the head bone is not assigned.
        public float GazeDirection
        {
            get
            {
                if (pepperController == null) return 0f;
                Transform gazeSource = headOrGazeBone ?? humanAgent?.Transform;
                if (gazeSource == null) return 0f;
                Vector3 gazeForward = gazeSource.forward;
                Vector3 toPepper = (pepperController.transform.position - gazeSource.position).normalized;
                float angle = Vector3.SignedAngle(gazeForward, toPepper, Vector3.up);
                return Mathf.Clamp(angle / 180f, -1f, 1f);
            }
        }

        // Unity lifecycle
        private void Awake()
        {
            if (pepperController != null)
            {
                initialPepperPosition = pepperController.transform.position;
                initialPepperRotation = pepperController.transform.rotation;
            }
            SetupHumanAgent();
        }

        void Start()
        {
            AutoFillReferences();
    
            SubscribeToPepperEvents();
            InitializeRewardProvider();

            if (pepperAgent != null)
                pepperAgent.SetCommunicationManager(this);

            // Pass the master seed to NPC before first episode
            SetMasterSeed(masterSeed);

            totalTasksPerformed = 0;
        }

        private void OnDestroy()
        {
            UnsubscribeFromPepperEvents();
            UnsubscribeFromHumanEvents();

            if (rewardProvider != null)
                rewardProvider.OnReward.RemoveListener(OnRewardReceived);
        }

        private void Update()
        {
            // Non-VR keyboard shortcuts for manual testing / heuristic mode
            if (!isVRMode)
                HandleDebugKeys();
        }


        /// Set the master seed for this entire run.
        /// Called from Python via config file parameter.
        public void SetMasterSeed(int seed)
        {
            masterSeed = seed;
            if (!isVRMode && humanAgent is NPCController npc)
            {
                npc.SetMasterSeed(seed);
                Debug.Log($"[CommManager] Master seed set to: {seed}");
            }
        }

        /// Called at the beginning of each episode.
        public void OnEpisodeBegin()
        {
            currentEpisodeNumber++;
            if (!isVRMode && humanAgent is NPCController npc)
            {
                npc.OnEpisodeBegin(currentEpisodeNumber);
            }
        }

        //  Reward Provider Setup
        private void InitializeRewardProvider()
        {
            switch (experimentMode)
            {
                case ExperimentMode.Baseline: // Autonomous Baseline
                    var autoProvider = GetComponent<AutonomousRewardProvider>();
                    if (autoProvider == null)
                        autoProvider = gameObject.AddComponent<AutonomousRewardProvider>();
                    rewardProvider = autoProvider;
                    Debug.Log("[CommManager] Mode 0: Autonomous Baseline - Robot gets automatic rewards");
                    break;

                case ExperimentMode.Keyboard: // Keyboard IRL
                    var keyboardProvider = GetComponent<KeyboardRewardProvider>();
                    if (keyboardProvider == null)
                        keyboardProvider = gameObject.AddComponent<KeyboardRewardProvider>();
                    rewardProvider = keyboardProvider;
                    Debug.Log("[CommManager] Mode 1: Keyboard IRL - Use up-arrow for +1, down-arrow for -1");
                    break;

                case ExperimentMode.VR: // VR Multimodal IRL
                    var vrProvider = GetComponent<VRMultimodalReward>();
                    if (vrProvider == null)
                        vrProvider = gameObject.AddComponent<VRMultimodalReward>();
                    rewardProvider = vrProvider;
                    Debug.Log("[CommManager] Mode 2: VR Multimodal IRL - Buttons + Voice + Head Gestures");
                    break;

                default:
                    Debug.LogWarning(
                        $"[CommManager] Unknown experiment mode: {experimentMode}, defaulting to Autonomous");
                    var defaultProvider = GetComponent<AutonomousRewardProvider>();
                    if (defaultProvider == null)
                        defaultProvider = gameObject.AddComponent<AutonomousRewardProvider>();
                    rewardProvider = defaultProvider;
                    break;
            }

            if (rewardProvider != null)
            {
                rewardProvider.IsEnabled = true;
                rewardProvider.OnReward.AddListener(OnRewardReceived);
            }
        }

        private void OnRewardReceived(float reward)
        {
            if (pepperAgent != null)
            {
                pepperAgent.AddReward(reward);
                currentFeedback = reward;
                Debug.Log($"[CommManager] Reward delivered to PepperAgent: {reward:+#;-#;0}");
            }
        }

        //  Setup Methods

        private void AutoFillReferences()
        {
            if (pepperController == null)
                pepperController = FindFirstObjectByType<PepperController>();

            if (pepperAgent == null)
                pepperAgent = FindFirstObjectByType<PepperAgent>();
        }

        private void SetupHumanAgent()
        {
            var vrPerson = FindFirstObjectByType<VRPersonController>();
            if (vrPerson != null)
            {
                isVRMode = true;
                humanAgent = vrPerson;
                vrPerson.SetPepperAgent(pepperAgent);
                Debug.Log("[CommManager] VR mode - human gives rewards via multimodal feedback");
            }
            else
            {
                var npc = FindFirstObjectByType<NPCController>();
                if (npc != null)
                {
                    isVRMode = false;
                    humanAgent = npc;
                    initialNPCPosition = npc.transform.position;
                    initialNPCRotation = npc.transform.rotation;
                    Debug.Log("[CommManager] NPC mode - autonomous rewards active");
                }
                else
                {
                    Debug.LogError("[CommManager] No IHumanAgent found in scene!");
                }
            }

            SubscribeToHumanEvents();
        }

        // Event wiring
        private void SubscribeToPepperEvents()
        {
            if (pepperController == null) return;
            pepperController.onActionPerformed.AddListener(OnPepperActionPerformed);
            pepperController.onStateChanged.AddListener(OnPepperStateChanged);
        }

        private void UnsubscribeFromPepperEvents()
        {
            if (pepperController == null) return;
            pepperController.onActionPerformed.RemoveListener(OnPepperActionPerformed);
            pepperController.onStateChanged.RemoveListener(OnPepperStateChanged);
        }

        private void SubscribeToHumanEvents()
        {
            if (humanAgent == null) return;
            humanAgent.OnStateChanged.AddListener(OnHumanStateChanged);
            humanAgent.OnTaskChanged.AddListener(OnHumanTaskChanged);
        }

        private void UnsubscribeFromHumanEvents()
        {
            if (humanAgent == null) return;
            humanAgent.OnStateChanged.RemoveListener(OnHumanStateChanged);
            humanAgent.OnTaskChanged.RemoveListener(OnHumanTaskChanged);
        }

        // Action execution
        //Called by PepperAgent every decision step.
        public void ExecutePepperAction(PepperController.AgentAction action)
        {
            if (pepperController == null) return;

            // Prevent stacking actions if the robot is already performing one.
            if (pepperController.CurrentState != PepperController.PepperState.Idle &&
                action != PepperController.AgentAction.DoNothing)
            {
                return;
            }

            pepperController.ExecuteAction(action);
            bool isCorrect = EvaluateReward(action);

            if (isCorrect)
            {
                hasActedForCurrentTask = true;
                Log($"Success with {action}. Locking until next task.");
            }
            else
            {
                Log($"Failed with {action}. Preparing retry...");
                StartCoroutine(RetryRoutine());
            }
        }

        // Reward Logic 

        private bool EvaluateReward(PepperController.AgentAction action)
        {
            // In VR mode, rewards come directly from the human via multimodal feedback.
            if (isVRMode)
            {
                Log("VR mode - waiting for human feedback");
                return true;
            }

            if (pepperAgent == null || humanAgent == null) return true;

            // NPC mode - use autonomous reward provider for baseline.
            if (experimentMode == ExperimentMode.Baseline && rewardProvider is AutonomousRewardProvider autoProvider)
            {
                int taskId = CurrentNPCTaskId;
                float distance = CurrentDistance;
                float rewardResult = autoProvider.CheckReward(taskId, action, distance);
                return rewardResult > 0;
            }

            if (experimentMode == ExperimentMode.Keyboard && rewardProvider is KeyboardRewardProvider)
            {
                Log("Keyboard IRL mode - waiting for key presses");
            }
            else if (experimentMode == ExperimentMode.VR && rewardProvider is VRMultimodalReward)
            {
                Log("VR Multimodal mode - waiting for human feedback");
            }

            return true;
        }

        private bool isRetrying = false;

        private IEnumerator RetryRoutine()
        {
            if (isRetrying) yield break;
            isRetrying = true;

            float timeout = 5f;
            float elapsed = 0f;

            while (pepperController.CurrentState != PepperController.PepperState.Idle)
            {
                elapsed += Time.deltaTime;
                if (elapsed >= timeout)
                {
                    Log("RetryRoutine timed out - forcing Idle.");
                    pepperController.CurrentState = PepperController.PepperState.Idle;
                    break;
                }

                yield return null;
            }

            isRetrying = false;

            if (!hasActedForCurrentTask)
            {
                Log("Retrying decision...");
                pepperAgent?.RequestDecision();
            }
        }

        // Handshake Logic

        private void OnPepperActionPerformed(PepperController.AgentAction action)
        {
            if (action == PepperController.AgentAction.HandShake
                && canHandshake
                && CurrentDistance <= interactionDistanceThreshold)
            {
                StartHandshake();
            }
        }

        private void OnPepperStateChanged(PepperController.PepperState state)
        {
            if (isHandshakeInProgress && state == PepperController.PepperState.Idle)
                ResetHandshake();
        }

        private void OnHumanStateChanged(NPCController.NPCState state)
        {
            if (isHandshakeInProgress && state == NPCController.NPCState.PerformingTask)
                CompleteHandshake();
        }

        [Header("Observation Timing")] [SerializeField]
        private float
            observationValueNoise = 0.05f; // small noise added to wrist/arm values so agent generalises to real humans

        private void OnHumanTaskChanged(NPCTask task)
        {
            taskRewardedThisEvent = false;
            hasActedForCurrentTask = false;
            lastRewardedTaskId = task?.id ?? -1;
            lastRewardedNPCState = humanAgent?.CurrentState ?? NPCController.NPCState.Idle;
            Log($"New task started: {task?.taskName ?? "none"} (id={lastRewardedTaskId})");
            totalTasksPerformed++;
            if (pepperController.CurrentState != PepperController.PepperState.Idle) return;
            StartCoroutine(RequestDecisionDelayed());
        }

        private IEnumerator RequestDecisionDelayed()
        {
            NPCTask task = humanAgent?.CurrentTask;
            float delay = task != null
                ? task.observationDelay + Random.Range(-task.observationDelayNoise, task.observationDelayNoise)
                : 1f;
            yield return new WaitForSeconds(Mathf.Max(0f, delay));
            pepperAgent?.RequestDecision();
        }

        private void StartHandshake()
        {
            isHandshakeInProgress = true;
            canHandshake = false;
            StartCoroutine(HandshakeCooldownCoroutine());
        }

        private void CompleteHandshake()
        {
            ResetHandshake();
            float taskDuration = humanAgent?.CurrentTask?.taskDuration ?? 0f;
            StartCoroutine(EndEpisodeAfterDelay(taskDuration, "Handshake success"));
        }

        private IEnumerator EndEpisodeAfterDelay(float delay, string reason)
        {
            yield return new WaitForSeconds(delay);
            pepperAgent?.EndEpisodeWithReason(reason);
        }

        public void ResetHandshake()
        {
            isHandshakeInProgress = false;
            canHandshake = true;
        }

        private IEnumerator HandshakeCooldownCoroutine()
        {
            yield return new WaitForSeconds(handshakeCooldown);
            canHandshake = true;
        }

        // Episode Reset
        public void ResetSimulation()
        {
            ResetHandshake();
            ResetRewardGate();
            ResetPepper();
            ResetHuman();

            Debug.Log($"Total Tasks Performed in episode: {totalTasksPerformed}");
            totalTasksPerformed = 0;
            hasActedForCurrentTask = false;
            rewardProvider?.Reset();
            isRetrying = false;
        }

        private void ResetPepper()
        {
            if (pepperController == null) return;
            pepperController.CurrentState = PepperController.PepperState.Idle;
            pepperController.StopLooking();
            pepperController.transform.SetPositionAndRotation(initialPepperPosition, initialPepperRotation);
        }

        private void ResetHuman()
        {
            if (humanAgent == null) return;

            humanAgent.StopMovement();
            humanAgent.ClearCurrentTask();

            if (!isVRMode && humanAgent is NPCController npc)
            {
                npc.transform.SetPositionAndRotation(initialNPCPosition, initialNPCRotation);
            }

            humanAgent.ForceStartTask();
        }

        public void ResetRewardGate()
        {
            lastRewardedTaskId = -1;
            lastRewardedNPCState = NPCController.NPCState.Idle;
            taskRewardedThisEvent = false;
        }

        // Debug Keyboard (Non-VR only)

        private void HandleDebugKeys()
        {
            if (Input.GetKeyDown(KeyCode.I)) ExecutePepperAction(PepperController.AgentAction.Talk);
            if (Input.GetKeyDown(KeyCode.W)) ExecutePepperAction(PepperController.AgentAction.DoNothing);
            if (Input.GetKeyDown(KeyCode.H)) ExecutePepperAction(PepperController.AgentAction.Look);
            if (Input.GetKeyDown(KeyCode.L)) ExecutePepperAction(PepperController.AgentAction.Wave);
            if (Input.GetKeyDown(KeyCode.R)) ExecutePepperAction(PepperController.AgentAction.HandShake);

            if (experimentMode != ExperimentMode.Keyboard && experimentMode != ExperimentMode.VR)
            {
                if (Input.GetKeyDown(KeyCode.UpArrow))
                {
                    pepperAgent?.AddReward(keyboardRewardAmount);
                    currentFeedback = keyboardRewardAmount;
                    Log($"Manual reward: +{keyboardRewardAmount}");
                }

                if (Input.GetKeyDown(KeyCode.DownArrow))
                {
                    pepperAgent?.AddReward(-keyboardRewardAmount);
                    currentFeedback = -keyboardRewardAmount;
                    Log($"Manual reward: -{keyboardRewardAmount}");
                }
            }
        }

        //  Helpers

        private float DistanceBetweenAgents()
        {
            if (pepperController == null || humanAgent?.Transform == null)
                return float.MaxValue;

            return Vector3.Distance(pepperController.transform.position, humanAgent.Transform.position);
        }

        private void Log(string msg)
        {
#if UNITY_EDITOR
            if (logInteractions)
                Debug.Log($"[CommManager] {msg}");
#endif
        }
    }
}