using System.Collections;
using Agents.NPC;
using Agents.Robot;
using Tasks;
using UnityEngine;

namespace Managers
{
    public class CommunicationManager : MonoBehaviour
    {
        [Header("Agent References")] [SerializeField]
        private PepperController pepperController;

        [SerializeField] private NPCController npcController;

        [Header("Interaction Settings")] [SerializeField]
        private float interactionDistanceThreshold = 3.0f;

        [SerializeField] private float handshakeCooldown = 3.0f;

        [Header("Keyboard Controls")] [SerializeField]
        private KeyCode keyWait = KeyCode.I;

        [SerializeField] private KeyCode keyDoNothing = KeyCode.W;
        [SerializeField] private KeyCode keyLook = KeyCode.H;
        [SerializeField] private KeyCode keyWave = KeyCode.L;
        [SerializeField] private KeyCode keyHandshake = KeyCode.R;

        [Header("AI / ML-Agents")] [SerializeField]
        private PepperAgent pepperAgent;

        [Header("Debug")] [SerializeField] private bool logInteractions = true;

        private bool isHandshakeInProgress = false;
        private bool canHandshake = true;
        private Coroutine currentHandshakeCoroutine;

        public PepperController.PepperState CurrentPepperState =>
            pepperController != null ? pepperController.CurrentState : PepperController.PepperState.Idle;

        public NPCController.NPCState CurrentNPCState =>
            npcController != null ? npcController.CurrentState : NPCController.NPCState.Idle;

        public float CurrentDistance => GetDistanceBetweenAgents();

        public bool IsHandshakeInProgress => isHandshakeInProgress;
        public bool CanHandshake => canHandshake;

        public PepperController PepperController => pepperController;
        public NPCController NpcController => npcController;

        private Vector3 initialPepperPosition;
        private Quaternion initialPepperRotation;
        private Vector3 initialNPCPosition;
        private Quaternion initialNPCRotation;

        public int CurrentNPCTaskId
        {
            get
            {
                if (npcController == null || npcController.CurrentTask == null)
                    return 0; // no task = 0
                return npcController.CurrentTask.id;
            }
        }

        private void Awake()
        {
            if (pepperController != null)
            {
                initialPepperPosition = pepperController.transform.position;
                initialPepperRotation = pepperController.transform.rotation;
            }

            if (npcController != null)
            {
                initialNPCPosition = npcController.transform.position;
                initialNPCRotation = npcController.transform.rotation;
            }
        }

        private void Start()
        {
            ValidateReferences();
            SubscribeToEvents();

            if (pepperAgent == null)
            {
                pepperAgent = FindFirstObjectByType<PepperAgent>();
                if (pepperAgent != null)
                    pepperAgent.SetCommunicationManager(this);
            }
        }

        private void OnDestroy()
        {
            UnsubscribeFromEvents();
        }

        private void Update()
        {
            HandleKeyboardInput();

            if (logInteractions)
            {
                // Debug.Log($"[Comm Status] Pepper: {CurrentPepperState,-12} | NPC: {CurrentNPCState,-12} | Dist: {CurrentDistance:F1}m | HS: {(isHandshakeInProgress ? "active" : "ready")}");
            }
        }

        private void ValidateReferences()
        {
            if (pepperController == null)
                pepperController = FindFirstObjectByType<PepperController>();

            if (npcController == null)
                npcController = FindFirstObjectByType<NPCController>();
        }

        private void SubscribeToEvents()
        {
            if (pepperController != null)
            {
                pepperController.onActionPerformed.AddListener(HandlePepperAction);
                pepperController.onStateChanged.AddListener(HandlePepperStateChange);
            }

            if (npcController != null)
            {
                npcController.onStateChanged.AddListener(HandleNPCStateChange);
                npcController.onTaskChanged.AddListener(HandleNPCTaskChange);
            }
        }

        private void UnsubscribeFromEvents()
        {
            if (pepperController != null)
            {
                pepperController.onActionPerformed.RemoveListener(HandlePepperAction);
                pepperController.onStateChanged.RemoveListener(HandlePepperStateChange);
            }

            if (npcController != null)
            {
                npcController.onStateChanged.RemoveListener(HandleNPCStateChange);
                npcController.onTaskChanged.RemoveListener(HandleNPCTaskChange);
            }
        }

        #region Keyboard Input

        private void HandleKeyboardInput()
        {
            if (Input.GetKeyDown(keyWait) || Input.GetKeyDown(KeyCode.Alpha0))
                ExecutePepperAction(PepperController.AgentAction.Wait);
            else if (Input.GetKeyDown(keyDoNothing) || Input.GetKeyDown(KeyCode.Alpha1))
                ExecutePepperAction(PepperController.AgentAction.DoNothing);
            else if (Input.GetKeyDown(keyLook) || Input.GetKeyDown(KeyCode.Alpha2))
                ExecutePepperAction(PepperController.AgentAction.Look);
            else if (Input.GetKeyDown(keyWave) || Input.GetKeyDown(KeyCode.Alpha3))
                ExecutePepperAction(PepperController.AgentAction.Wave);
            else if (Input.GetKeyDown(keyHandshake) || Input.GetKeyDown(KeyCode.Alpha4))
                ExecutePepperAction(PepperController.AgentAction.HandShake);
        }

        #endregion

        #region Action Execution

        public void ExecutePepperAction(PepperController.AgentAction action)
        {
            if (pepperController == null)
                return;

            if (action == PepperController.AgentAction.HandShake && !canHandshake)
            {
                LogInteraction("Handshake blocked - on cooldown or too far");
                return;
            }

            pepperController.ExecuteAction(action);

            // TASK-BASED RL REWARD
            EvaluatePepperActionReward(action);
        }

        #endregion

        #region Reward Logic

        private void EvaluatePepperActionReward(PepperController.AgentAction action)
        {
            if (pepperAgent == null)
                return;

            int taskId = CurrentNPCTaskId;
            float distance = CurrentDistance;

            bool correct = false;
            float rewardValue = 0f;

            switch (taskId)
            {
                case 2: // Handshake
                    correct = action == PepperController.AgentAction.HandShake;
                    rewardValue = correct ? 2.0f : -0.5f;
                    break;

                case 7: // WaveFromDistance
                    correct = action == PepperController.AgentAction.Wave;
                    rewardValue = correct ? 1.5f : -0.3f;
                    break;

                case 6: // TalkInMiddle
                    correct = action == PepperController.AgentAction.Wait;
                    rewardValue = correct ? 1.0f : -0.2f;
                    break;

                default:
                    if (distance <= 3.0f)
                        correct = action == PepperController.AgentAction.Look;
                    else
                        correct = action == PepperController.AgentAction.DoNothing;

                    rewardValue = correct ? 0.5f : -0.1f;
                    break;
            }

            pepperAgent.AddReward(rewardValue);
            if (logInteractions)
                Debug.Log($"[Reward {rewardValue}] Task {taskId} | Action {action}");
        }

        #endregion

        #region Event Handlers

        private void HandlePepperAction(PepperController.AgentAction action)
        {
            float distance = CurrentDistance;

            if (action == PepperController.AgentAction.HandShake &&
                canHandshake && distance <= interactionDistanceThreshold)
            {
                StartHandshakeInteraction();
            }
        }

        private void HandlePepperStateChange(PepperController.PepperState state)
        {
            if (state == PepperController.PepperState.Idle && isHandshakeInProgress)
                ResetHandshake();
        }

        private void HandleNPCStateChange(NPCController.NPCState state)
        {
            if (isHandshakeInProgress && state == NPCController.NPCState.PerformingTask)
                CompleteHandshake();
        }

        private void HandleNPCTaskChange(NPCTask task)
        {
            // Task changes are handled implicitly by reward logic
        }

        #endregion

        #region Handshake Logic

        private void StartHandshakeInteraction()
        {
            isHandshakeInProgress = true;
            canHandshake = false;

            if (currentHandshakeCoroutine != null)
                StopCoroutine(currentHandshakeCoroutine);

            currentHandshakeCoroutine = StartCoroutine(HandshakeCooldown());
        }

        private void CompleteHandshake()
        {
            if (pepperAgent != null)
                pepperAgent.EndEpisodeSuccess("Handshake success");

            ResetHandshake();
        }

        public void ResetHandshake()
        {
            isHandshakeInProgress = false;
            canHandshake = true;
            LogInteraction("Handshake reset");
        }

        public void ResetSimulation()
        {
            // 1. Reset handshake / interaction state
            ResetHandshake();

            // 2. Reset Pepper look-at / animation state
            if (pepperController != null)
            {
                pepperController.CurrentState = PepperController.PepperState.Idle; // force idle

                pepperController.StopLooking();

                // teleport back to start position (prevents getting stuck far away)
                pepperController.transform.SetPositionAndRotation(
                    initialPepperPosition,
                    initialPepperRotation
                );
            }

            // 3. Reset NPC task / movement / state
            if (npcController != null)
            {
                // Force NPC back to a predictable state
                npcController.CurrentState = NPCController.NPCState.WaitingBetweenTasks;

                npcController.ClearCurrentTask();

                // reset position (especially useful if NPC wanders far)
                npcController.transform.SetPositionAndRotation(
                    initialNPCPosition,
                    initialNPCRotation
                );

                // stop NavMeshAgent movement immediately
                npcController.StopNavMeshAgent();
            }
        }

        private IEnumerator HandshakeCooldown()
        {
            yield return new WaitForSeconds(handshakeCooldown);
            canHandshake = true;
        }

        #endregion

        #region Helpers

        private float GetDistanceBetweenAgents()
        {
            if (pepperController == null || npcController == null)
                return float.MaxValue;

            return Vector3.Distance(
                pepperController.transform.position,
                npcController.transform.position
            );
        }

        private void LogInteraction(string message)
        {
            if (logInteractions)
            {
                // Debug.Log($"[Interaction] {message}");
            }
        }

        #endregion
    }
}