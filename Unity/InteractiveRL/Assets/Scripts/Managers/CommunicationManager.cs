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
        private KeyCode keyWait = KeyCode.I; // or Alpha0

        [SerializeField] private KeyCode keyDoNothing = KeyCode.W; // or Alpha1
        [SerializeField] private KeyCode keyLook = KeyCode.H; // or Alpha2
        [SerializeField] private KeyCode keyWave = KeyCode.L; // or Alpha3
        [SerializeField] private KeyCode keyHandshake = KeyCode.R; // or Alpha4

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

        private void Awake()
        {
            // Cache starting transforms (do this once)
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
                {
                    pepperAgent.SetCommunicationManager(this);
                }
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
            {
                pepperController = FindFirstObjectByType<PepperController>();
                if (pepperController != null && logInteractions)
                    Debug.Log("[CommunicationManager] Found PepperController in scene");
            }

            if (npcController == null)
            {
                npcController = FindFirstObjectByType<NPCController>();
                if (npcController != null && logInteractions)
                    Debug.Log("[CommunicationManager] Found NPCController in scene");
            }
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
            {
                ExecutePepperAction(PepperController.AgentAction.Wait);
            }
            else if (Input.GetKeyDown(keyDoNothing) || Input.GetKeyDown(KeyCode.Alpha1))
            {
                ExecutePepperAction(PepperController.AgentAction.DoNothing);
            }
            else if (Input.GetKeyDown(keyLook) || Input.GetKeyDown(KeyCode.Alpha2))
            {
                ExecutePepperAction(PepperController.AgentAction.Look);
            }
            else if (Input.GetKeyDown(keyWave) || Input.GetKeyDown(KeyCode.Alpha3))
            {
                ExecutePepperAction(PepperController.AgentAction.Wave);
            }
            else if (Input.GetKeyDown(keyHandshake) || Input.GetKeyDown(KeyCode.Alpha4))
            {
                ExecutePepperAction(PepperController.AgentAction.HandShake);
            }
        }

        #endregion

        #region Central Action Entry Point (used by keyboard + AI + future VR)

        public void ExecutePepperAction(PepperController.AgentAction action)
        {
            if (pepperController == null)
            {
                Debug.LogWarning("[Comm] Cannot execute action — PepperController is null");
                return;
            }

            // block or modify actions based on state (example)
            if (action == PepperController.AgentAction.HandShake && !CanHandshake)
            {
                LogInteraction("Handshake blocked - on cooldown or too far");
                return;
            }

            LogInteraction($"Executing action: {action}");
            pepperController.ExecuteAction(action);

            // sparse environment reward feedback for ML-Agents
            if (pepperAgent != null)
            {
                if (action == PepperController.AgentAction.HandShake && CurrentDistance <= 2.0f)
                {
                    pepperAgent.AddReward(1.5f);
                }
                else if (action == PepperController.AgentAction.Wave && CurrentDistance <= interactionDistanceThreshold)
                {
                    pepperAgent.AddReward(0.4f);
                }
            }
        }

        #endregion

        #region Event Handlers

        private void HandlePepperAction(PepperController.AgentAction action)
        {
            //if (logInteractions)
            //    Debug.Log($"[Pepper Action] {action}");

            float distance = CurrentDistance;

            switch (action)
            {
                case PepperController.AgentAction.Wave:
                    if (distance <= interactionDistanceThreshold)
                    {
                        NotifyNPCAboutPepperAction("wave");
                        LogInteraction("Pepper waved at NPC");
                    }

                    break;

                case PepperController.AgentAction.HandShake:
                    if (canHandshake && distance <= interactionDistanceThreshold)
                    {
                        StartHandshakeInteraction();
                    }
                    else if (!canHandshake)
                    {
                        LogInteraction("Handshake is on cooldown");
                    }
                    else
                    {
                        LogInteraction($"NPC too far for handshake (Dist: {distance:F1}m)");
                    }

                    break;

                case PepperController.AgentAction.Look:
                    if (distance <= interactionDistanceThreshold * 1.5f)
                    {
                        LogInteraction("Pepper is looking at NPC");
                    }

                    break;
            }
        }

        private void HandlePepperStateChange(PepperController.PepperState state)
        {
            if (state == PepperController.PepperState.Idle && isHandshakeInProgress)
            {
                ResetHandshake();
            }
        }

        private void HandleNPCStateChange(NPCController.NPCState state)
        {
            if (logInteractions)
                Debug.Log($"[NPC State] {state}");

            if (isHandshakeInProgress && state == NPCController.NPCState.PerformingTask)
            {
                CompleteHandshake();
            }
        }

        private void HandleNPCTaskChange(NPCTask task)
        {
            if (task == null) return;

            if (logInteractions)
               // Debug.Log($"[NPC Task] Started: {task.taskName}");

            HandleNPCToPepperReaction(task);
        }

        #endregion

        #region Interaction Logic

        private void StartHandshakeInteraction()
        {
            isHandshakeInProgress = true;
            canHandshake = false;

            LogInteraction("Pepper initiated handshake with NPC");

            if (currentHandshakeCoroutine != null)
                StopCoroutine(currentHandshakeCoroutine);
            currentHandshakeCoroutine = StartCoroutine(HandshakeCooldown());

            NotifyNPCAboutPepperAction("handshake");
        }

        private void CompleteHandshake()
        {
            if (isHandshakeInProgress)
            {
                LogInteraction("Handshake completed successfully");

                // Optional positive reward signal for ML-Agents
                if (pepperAgent != null){
                    pepperAgent.EndEpisodeSuccess("Handshake success");
                }
                ResetHandshake();
            }
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
            LogInteraction("Handshake cooldown complete");
        }

        private void HandleNPCToPepperReaction(NPCTask task)
        {
            string taskName = task.taskName.ToLower();

            if (taskName.Contains("wave") || taskName.Contains("greet"))
            {
                if (pepperController.CurrentState == PepperController.PepperState.Looking)
                {
                    LogInteraction("NPC is waving back at Pepper");
                }
            }
            else if (taskName.Contains("handshake") || taskName.Contains("shake"))
            {
                if (!isHandshakeInProgress && canHandshake)
                {
                    LogInteraction("NPC wants to handshake");
                    // Future: could auto-trigger Pepper response here
                }
            }
        }

        private void NotifyNPCAboutPepperAction(string action)
        {
            LogInteraction($"NPC notified about Pepper's {action} action");
            // Future extension point: set NPC state, trigger animation, inject task, etc.
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

        #region Public Utility Methods

        public void SetAgents(PepperController pepper, NPCController npc)
        {
            UnsubscribeFromEvents();

            pepperController = pepper;
            npcController = npc;

            SubscribeToEvents();

            if (logInteractions)
                Debug.Log("[CommunicationManager] Agents updated");
        }

        public bool IsInteractionAvailable()
        {
            return canHandshake && CurrentDistance <= interactionDistanceThreshold;
        }

        public string GetInteractionStatus()
        {
            float distance = CurrentDistance;
            return $"Distance: {distance:F1}m | Handshake Ready: {canHandshake} | In Progress: {isHandshakeInProgress}";
        }

        #endregion
    }
}