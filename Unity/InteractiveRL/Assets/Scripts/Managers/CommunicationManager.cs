using System.Collections;
using Agents.NPC;
using Agents.Robot;
using Tasks;
using UnityEngine;
using UnityEngine.Events;

public class CommunicationManager : MonoBehaviour
{
    [Header("Agent References")]
    [SerializeField] private PepperController pepperController;
    [SerializeField] private NPCController npcController;

    [Header("Interaction Settings")]
    [SerializeField] private float interactionDistanceThreshold = 3.0f;
    [SerializeField] private float handshakeCooldown = 3.0f;

    [Header("Debug")]
    [SerializeField] private bool logInteractions = true;

    // Internal state
    private bool isHandshakeInProgress = false;
    private bool canHandshake = true;
    private Coroutine currentHandshakeCoroutine;
    public PepperController PepperController => pepperController;
    public NPCController NpcController => npcController;
    private bool keysEnabled = true;
    private void Start()
    {
        ValidateReferences();
        SubscribeToEvents();
    }

    private void OnDestroy()
    {
        UnsubscribeFromEvents();
    }

    void Update()
    {
        HandleKeyboardInput();
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
        // Subscribe to Pepper events
        if (pepperController != null)
        {
            pepperController.onActionPerformed.AddListener(HandlePepperAction);
            pepperController.onStateChanged.AddListener(HandlePepperStateChange);
        }

        // Subscribe to NPC events
        if (npcController != null)
        {
            npcController.onStateChanged.AddListener(HandleNPCStateChange);
            npcController.onTaskChanged.AddListener(HandleNPCTaskChange);
        }
    }

    private void UnsubscribeFromEvents()
    {
        // Unsubscribe from Pepper events
        if (pepperController != null)
        {
            pepperController.onActionPerformed.RemoveListener(HandlePepperAction);
            pepperController.onStateChanged.RemoveListener(HandlePepperStateChange);
        }

        // Unsubscribe from NPC events
        if (npcController != null)
        {
            npcController.onStateChanged.RemoveListener(HandleNPCStateChange);
            npcController.onTaskChanged.RemoveListener(HandleNPCTaskChange);
        }
    }

    #region Event Handlers

    private void HandlePepperAction(PepperController.AgentAction action)
    {
        if (logInteractions)
            Debug.Log($"[Pepper Action] {action}");

        float distance = GetDistanceBetweenAgents();

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
                    LogInteraction($"NPC is too far for handshake (Distance: {distance:F1}m)");
                }
                break;

            case PepperController.AgentAction.Look:
                // NPC might react to being looked at
                if (distance <= interactionDistanceThreshold * 1.5f)
                {
                    LogInteraction("Pepper is looking at NPC");
                }
                break;
        }
    }

    private void HandlePepperStateChange(PepperController.PepperState state)
    {
        // Reset handshake if pepper goes idle during handshake
        if (state == PepperController.PepperState.Idle && isHandshakeInProgress)
        {
            ResetHandshake();
        }
    }

    private void HandleNPCStateChange(NPCController.NPCState state)
    {
        if (logInteractions)
            Debug.Log($"[NPC State] {state}");

        // Handle handshake completion
        if (isHandshakeInProgress && state == NPCController.NPCState.PerformingTask)
        {
            CompleteHandshake();
        }
    }

    private void HandleNPCTaskChange(NPCTask task)
    {
        if (task == null) return;

        if (logInteractions)
            Debug.Log($"[NPC Task] Started: {task.taskName}");

        // Check if Pepper should react to NPC's task
        HandleNPCToPepperReaction(task);
    }

    #endregion

    #region Interaction Logic

    private void StartHandshakeInteraction()
    {
        isHandshakeInProgress = true;
        canHandshake = false;

        LogInteraction("Pepper initiated handshake with NPC");

        // Start cooldown
        if (currentHandshakeCoroutine != null)
            StopCoroutine(currentHandshakeCoroutine);
        currentHandshakeCoroutine = StartCoroutine(HandshakeCooldown());

        // Trigger NPC to respond (if NPC has handshake capability)
        NotifyNPCAboutPepperAction("handshake");
    }

    private void CompleteHandshake()
    {
        if (isHandshakeInProgress)
        {
            LogInteraction("Handshake completed successfully");
            ResetHandshake();
        }
    }

    private void ResetHandshake()
    {
        isHandshakeInProgress = false;
        LogInteraction("Handshake reset");
    }

    private IEnumerator HandshakeCooldown()
    {
        yield return new WaitForSeconds(handshakeCooldown);
        canHandshake = true;
        LogInteraction("Handshake cooldown complete");
    }

    private void HandleNPCToPepperReaction(NPCTask task)
    {
        // Define which NPC tasks should get Pepper's attention
        string taskName = task.taskName.ToLower();

        if (taskName.Contains("wave") || taskName.Contains("greet"))
        {
            // Pepper could wave back if looking
            if (pepperController.CurrentState == PepperController.PepperState.Looking)
            {
                LogInteraction("NPC is waving back at Pepper");
            }
        }
        else if (taskName.Contains("handshake") || taskName.Contains("shake"))
        {
            // If NPC initiates handshake
            if (!isHandshakeInProgress && canHandshake)
            {
                LogInteraction("NPC wants to handshake");
                // Pepper could automatically respond here if desired
            }
        }
    }

    private void NotifyNPCAboutPepperAction(string action)
    {
        // This is where you'd trigger NPC responses
        // For now, we'll just log it
        LogInteraction($"NPC notified about Pepper's {action} action");

        // In a more advanced system, you could:
        // 1. Set a specific state on the NPC
        // 2. Trigger an animation
        // 3. Change NPC's current task
    }

    #endregion

    #region Helper Methods

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
            Debug.Log($"[Interaction] {message}");
        }
    }

    #endregion

    #region Public Methods

    public void SetAgents(PepperController pepper, NPCController npc)
    {
        // Unsubscribe from old agents
        UnsubscribeFromEvents();

        // Set new agents
        pepperController = pepper;
        npcController = npc;

        // Subscribe to new agents
        SubscribeToEvents();

        if (logInteractions)
            Debug.Log("[CommunicationManager] Agents updated");
    }
    
    private void HandleKeyboardInput()
    {
        if (!keysEnabled) return;

        if (Input.GetKeyDown(KeyCode.I) || Input.GetKeyDown(KeyCode.Alpha0))
        {
            RequestPepperAction(PepperController.AgentAction.Wait);
        }
        else if (Input.GetKeyDown(KeyCode.W) || Input.GetKeyDown(KeyCode.Alpha1))
        {
            RequestPepperAction(PepperController.AgentAction.DoNothing);
        }
        else if (Input.GetKeyDown(KeyCode.H) || Input.GetKeyDown(KeyCode.Alpha2))
        {
            RequestPepperAction(PepperController.AgentAction.Look);
        }
        else if (Input.GetKeyDown(KeyCode.L) || Input.GetKeyDown(KeyCode.Alpha3))
        {
            RequestPepperAction(PepperController.AgentAction.Wave);
        }
        else if (Input.GetKeyDown(KeyCode.R) || Input.GetKeyDown(KeyCode.Alpha4))
        {
            RequestPepperAction(PepperController.AgentAction.HandShake);
        }
    }
    
    private void RequestPepperAction(PepperController.AgentAction action)
    {
        if (pepperController == null)
        {
            Debug.LogWarning("[Comm] Cannot send action — PepperController reference is null");
            return;
        }

        LogInteraction($"Player requested Pepper action: {action}");
        pepperController.ExecuteAction(action);
    }

    public bool IsInteractionAvailable()
    {
        return canHandshake && GetDistanceBetweenAgents() <= interactionDistanceThreshold;
    }

    public string GetInteractionStatus()
    {
        float distance = GetDistanceBetweenAgents();
        return $"Distance: {distance:F1}m | Handshake Ready: {canHandshake} | In Progress: {isHandshakeInProgress}";
    }

    #endregion
    
    
}