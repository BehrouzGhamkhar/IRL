using System.Collections.Generic;
using Agents;
using Agents.NPC;
using Tasks;
using TMPro;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.InputSystem; // ← Use this instead of Valve.VR

namespace Agents.Human
{
    public class VRPersonController : MonoBehaviour, IHumanAgent
    {
        [Header("Available Tasks")]
        public List<NPCTask> availableTasks = new();

        [Header("VR Input Actions")]
        [Tooltip("Left controller secondary button (B/Y) to cycle tasks")]
        public InputActionReference cycleTaskAction;

        [Tooltip("Right controller trigger to start task")]
        public InputActionReference performTaskAction;

        [Header("Keyboard Fallback (Editor)")]
        public KeyCode cycleTaskKey = KeyCode.Tab;
        public KeyCode performTaskKey = KeyCode.Space;


        [Header("References")]
        [SerializeField] private Transform vrPlayerTransform;
        [SerializeField] private TextMeshProUGUI taskPromptText;

        [Header("State Events")]
        [SerializeField] private UnityEvent<NPCController.NPCState> onStateChanged = new();
        [SerializeField] private UnityEvent<NPCTask> onTaskChanged = new();

        public UnityEvent<NPCController.NPCState> OnStateChanged => onStateChanged;
        public UnityEvent<NPCTask> OnTaskChanged => onTaskChanged;
        public NPCTask CurrentTask => currentTask;
        public Transform Transform => vrPlayerTransform != null ? vrPlayerTransform : transform;
        public NPCController.NPCState CurrentState => currentState;

        private NPCTask currentTask;
        private int selectedTaskIndex;
        private float taskTimer;
        private bool hasStartedTask;
        private NPCController.NPCState currentState = NPCController.NPCState.Idle;
        private Agents.Robot.PepperAgent pepperAgent;

        private void OnEnable()
        {
            cycleTaskAction?.action.Enable();
            performTaskAction?.action.Enable();
        }

        private void OnDisable()
        {
            cycleTaskAction?.action.Disable();
            performTaskAction?.action.Disable();
        }

        private void Start()
        {
            if (vrPlayerTransform == null)
            {
                var xrOrigin = FindFirstObjectByType<Unity.XR.CoreUtils.XROrigin>();
                vrPlayerTransform = xrOrigin != null ? xrOrigin.transform : transform;

                if (xrOrigin == null)
                    Debug.LogWarning("[VRPerson] XROrigin not found - using own transform.");
            }

            if (availableTasks.Count == 0)
                Debug.Log("[VRPerson] No available tasks.");

            SetState(NPCController.NPCState.WaitingBetweenTasks);
            UpdateUI();
        }

        private void Update()
        {


            switch (currentState)
            {
                case NPCController.NPCState.WaitingBetweenTasks:
                    UpdateWaiting();
                    break;
                case NPCController.NPCState.PerformingTask:
                    UpdatePerforming();
                    break;
            }
        }

        // Input

        private bool GetCycleDown() =>
            (cycleTaskAction != null && cycleTaskAction.action.WasPressedThisFrame())
            || Input.GetKeyDown(cycleTaskKey);

        private bool GetPerformDown() =>
            (performTaskAction != null && performTaskAction.action.WasPressedThisFrame())
            || Input.GetKeyDown(performTaskKey);


        // State Updates

        private void UpdateWaiting()
        {
            if (availableTasks.Count == 0) return;

            if (GetCycleDown())
            {
                selectedTaskIndex = (selectedTaskIndex + 1) % availableTasks.Count;
                UpdateUI();
                Debug.Log($"[VRPerson] Selected: {availableTasks[selectedTaskIndex].taskName}");
            }

            if (GetPerformDown())
                StartTask(availableTasks[selectedTaskIndex]);
        }

        private void UpdatePerforming()
        {
            if (!hasStartedTask)
            {
                hasStartedTask = true;
                taskTimer = 0f;
                onTaskChanged?.Invoke(currentTask);
                Debug.Log($"[VRPerson] Started task: {currentTask.taskName}");
                UpdateUI();
            }

            taskTimer += Time.deltaTime;
            if (taskTimer >= currentTask.taskDuration)
                FinishTask();
        }

        // Task Control

        private void StartTask(NPCTask task)
        {
            currentTask = task;
            hasStartedTask = false;
            taskTimer = 0f;
            SetState(NPCController.NPCState.PerformingTask);
        }

        private void FinishTask()
        {
            Debug.Log($"[VRPerson] Finished: {currentTask?.taskName}");
            currentTask = null;
            hasStartedTask = false;
            taskTimer = 0f;
            SetState(NPCController.NPCState.WaitingBetweenTasks);
            UpdateUI();
        }

        private void SetState(NPCController.NPCState newState)
        {
            if (currentState == newState) return;
            currentState = newState;
            onStateChanged?.Invoke(currentState);
        }

        public void ForceStartTask()
        {
            currentTask = null;
            hasStartedTask = false;
            taskTimer = 0f;
            selectedTaskIndex = 0;
            SetState(NPCController.NPCState.WaitingBetweenTasks);
            UpdateUI();
        }

        public void ClearCurrentTask() => FinishTask();
        public void StopMovement() { }
        public void SetPepperAgent(Agents.Robot.PepperAgent agent) => pepperAgent = agent;

        private void UpdateUI()
        {
            if (taskPromptText == null) return;

            if (currentState == NPCController.NPCState.PerformingTask)
            {
                taskPromptText.text = $"Performing: {currentTask?.taskName}…";
                return;
            }

            if (availableTasks.Count == 0)
            {
                taskPromptText.text = "No tasks available";
                return;
            }

            var selected = availableTasks[selectedTaskIndex];
            taskPromptText.text =
                $"[B Button] Cycle   [Trigger] Start\n" +
                $"Task: {selected.taskName}  ({selectedTaskIndex + 1}/{availableTasks.Count})";
        }
    }
}