using System.Collections;
using Agents.NPC;
using Tasks;
using UnityEngine;
using UnityEngine.Events;

namespace Agents.Robot
{
    public class PepperController : MonoBehaviour
    {
        [SerializeField] private PepperAnimationController animationController;
        [SerializeField] private Transform headBone;
        [SerializeField] private float headRotationSpeed = 5f;
        [SerializeField] private float lookAtDuration = 3f;
    
        [Header("NPC Monitoring")]
        [SerializeField] public NPCController npcToMonitor; // Drag ONE NPC here in the inspector
        
        [Header("State Events")]
        public UnityEvent<PepperState> onStateChanged;
        public UnityEvent<AgentAction> onActionPerformed;
        public UnityEvent<NPCController.NPCState> onNPCStateChanged;
        public UnityEvent<NPCTask> onNPCTaskChanged;
    
        private Transform currentLookTarget;
        private float lookEndTime;
        private bool isLooking;
        private PepperState currentState = PepperState.Idle;
        private PepperState previousState;

        public enum AgentAction
        {
            DoNothing = 0,
            Wait = 1,
            Look = 2,
            Wave = 3,
            HandShake = 6
        };
        
        public enum PepperState
        {
            Idle,
            Looking,
            Waving,
            Handshaking,
            PerformingAction,
            MonitoringNPC
        }

        // Property to handle state changes with events
        public PepperState CurrentState
        {
            get { return currentState; }
            private set
            {
                if (currentState != value)
                {
                    previousState = currentState;
                    currentState = value;
                    
                    if (onStateChanged != null)
                    {
                        onStateChanged.Invoke(currentState);
                    }
                }
            }
        }

        void Start()
        {
            // Initialize events if they're null
            if (onStateChanged == null)
                onStateChanged = new UnityEvent<PepperState>();
            if (onActionPerformed == null)
                onActionPerformed = new UnityEvent<AgentAction>();
            if (onNPCStateChanged == null)
                onNPCStateChanged = new UnityEvent<NPCController.NPCState>();
            if (onNPCTaskChanged == null)
                onNPCTaskChanged = new UnityEvent<NPCTask>();

            if (animationController == null)
            {
                animationController = GetComponent<PepperAnimationController>();
                if (animationController == null)
                {
                    Debug.LogError("Robot Animator not found!");
                }
            }
        
            // Subscribe to NPC events
            SetupNPCEventListener();
            
            // Set initial state
            CurrentState = PepperState.Idle;
        }

        void OnDestroy()
        {
            // Clean up event listener when destroyed
            CleanupNPCEventListener();
        }

        void Update()
        {
            // Handle continuous look behavior
            if (isLooking && Time.time < lookEndTime && currentLookTarget != null)
            {
                CurrentState = PepperState.Looking;
                Vector3 lookDirection = currentLookTarget.position - headBone.position;
                Quaternion targetRotation = Quaternion.LookRotation(lookDirection);
                headBone.rotation = Quaternion.Slerp(
                    headBone.rotation, 
                    targetRotation, 
                    Time.deltaTime * headRotationSpeed
                );
            }
            else if (isLooking && Time.time >= lookEndTime)
            {
                isLooking = false;
                if (CurrentState == PepperState.Looking)
                {
                    CurrentState = PepperState.Idle;
                }
            }

            HandleKeyboardInput();
        }

        #region NPC Event Handling
    
        private void SetupNPCEventListener()
        {
            if (npcToMonitor != null)
            {
                // Subscribe to state changes
                if (npcToMonitor.onStateChanged != null)
                {
                    npcToMonitor.onStateChanged.AddListener(OnNPCStateChanged);
                }
            
                // Subscribe to task changes
                if (npcToMonitor.onTaskChanged != null)
                {
                    npcToMonitor.onTaskChanged.AddListener(OnNPCTaskChanged);
                }
            
                Debug.Log($"Pepper is now monitoring NPC: {npcToMonitor.gameObject.name}");
            
                // Log initial state
                LogNPCState(npcToMonitor.CurrentState);
                if (npcToMonitor.CurrentTask != null)
                {
                    LogNPCTask(npcToMonitor.CurrentTask);
                }
            }
            else
            {
                Debug.LogWarning("No NPC assigned to monitor. Drag an NPC into the 'npcToMonitor' field in the inspector.");
            }
        }
    
        private void CleanupNPCEventListener()
        {
            if (npcToMonitor != null)
            {
                if (npcToMonitor.onStateChanged != null)
                {
                    npcToMonitor.onStateChanged.RemoveListener(OnNPCStateChanged);
                }
            
                if (npcToMonitor.onTaskChanged != null)
                {
                    npcToMonitor.onTaskChanged.RemoveListener(OnNPCTaskChanged);
                }
            }
        }
    
        private void OnNPCStateChanged(NPCController.NPCState newState)
        {
            LogNPCState(newState);
            CurrentState = PepperState.MonitoringNPC;
            
            if (onNPCStateChanged != null)
            {
                onNPCStateChanged.Invoke(newState);
            }
        }
    
        private void OnNPCTaskChanged(NPCTask newTask)
        {
            LogNPCTask(newTask);
            CurrentState = PepperState.MonitoringNPC;
            
            if (onNPCTaskChanged != null)
            {
                onNPCTaskChanged.Invoke(newTask);
            }
        }
    
        private void LogNPCState(NPCController.NPCState state)
        {
            string npcName = npcToMonitor != null ? npcToMonitor.gameObject.name : "Unknown NPC";
            Debug.Log($"[Pepper Monitoring] '{npcName}' state changed to: {state}");
        }
    
        private void LogNPCTask(NPCTask task)
        {
            string npcName = npcToMonitor != null ? npcToMonitor.gameObject.name : "Unknown NPC";
        
            if (task != null)
            {
                Debug.Log($"[Pepper Monitoring] '{npcName}' started task: {task.taskName} (Target: {task.targetObjectName})");
            }
            else
            {
                Debug.Log($"[Pepper Monitoring] '{npcName}' has no current task");
            }
        }
    
        #endregion

        private void ExecuteAction(AgentAction rAction)
        {
            CurrentState = PepperState.PerformingAction;
            
            // Fire action performed event
            if (onActionPerformed != null)
            {
                onActionPerformed.Invoke(rAction);
            }
            
            switch (rAction)
            {
                case AgentAction.Wait:
                    ActionWait();
                    break;
                
                case AgentAction.Look:
                    ActionLook();
                    break;
                
                case AgentAction.Wave:
                    ActionLook(); // Look first
                    ActionWave();
                    break;
                
                case AgentAction.HandShake:
                    float tryHandShakeTime = 2.0f;
                    ActionLook(); // Look first
                    StartCoroutine(ActionHandshake(tryHandShakeTime));
                    break;
                
                case AgentAction.DoNothing:
                    // Intentionally blank
                    CurrentState = PepperState.Idle;
                    break;
                
                default:
                    Debug.LogWarning($"Unhandled action: {rAction}");
                    CurrentState = PepperState.Idle;
                    break;
            }
        }

        #region Action Implementations
    
        private void ActionWait()
        {
            animationController.PlayIdle();
            Debug.Log("[Pepper Action] Waiting");
            CurrentState = PepperState.Idle;
        }

        private void ActionLook()
        {
            var closestPerson = FindNearestPerson();
            currentLookTarget = closestPerson?.transform.Find("HeadPosition");

            if (currentLookTarget != null)
            {
                isLooking = true;
                lookEndTime = Time.time + lookAtDuration;
                Debug.Log("[Pepper Action] Looking at nearest person");
            }
            else
            {
                Debug.LogWarning("[Pepper Action] No person found to look at");
                CurrentState = PepperState.Idle;
            }
        }

        private void ActionWave()
        {
            animationController.PlayWave();
            Debug.Log("[Pepper Action] Waving");
            CurrentState = PepperState.Waving;
            StartCoroutine(ResetStateAfterAnimation(PepperState.Waving));
        }

        IEnumerator ActionHandshake(float delayTime)
        {
            animationController.PlayTryHandshake();
            Debug.Log("[Pepper Action] Attempting handshake");
            CurrentState = PepperState.Handshaking;
            
            var closestPerson = FindNearestPerson();
            yield return new WaitForSeconds(delayTime);
            if (closestPerson != null)
            {
                Vector3 targetPosition = closestPerson.position;
                if (Vector3.Distance(transform.position, targetPosition) < 2.0f)
                {
                    animationController.PlayHandshake();
                    Debug.Log("[Pepper Action] Handshake successful");
                }
                else
                {
                    Debug.LogWarning("[Pepper Action] Too far to handshake.");
                    animationController.PlayIdle();
                    CurrentState = PepperState.Idle;
                }
            }
            else
            {
                Debug.LogWarning("[Pepper Action] No person found to handshake with.");
                animationController.PlayIdle();
                CurrentState = PepperState.Idle;
            }
            
            StartCoroutine(ResetStateAfterAnimation(PepperState.Handshaking));
        }
        
        IEnumerator ResetStateAfterAnimation(PepperState stateToReset)
        {
            // Wait a moment for the animation to complete
            yield return new WaitForSeconds(1.5f);
            if (CurrentState == stateToReset)
            {
                CurrentState = PepperState.Idle;
            }
        }
    
        #endregion

        private Transform FindNearestPerson()
        {
            GameObject[] people = GameObject.FindGameObjectsWithTag("Person");
            float closestDistance = float.MaxValue;
            Transform closestPerson = null;

            if (people.Length == 0)
            {
                Debug.LogWarning("No person found to look at.");
                return null;
            }

            foreach (var person in people)
            {
                float distance = Vector3.Distance(transform.position, person.transform.position);
                if (distance < closestDistance)
                {
                    closestDistance = distance;
                    closestPerson = person.transform;
                }
            }
            return closestPerson;
        }
    
        private void HandleKeyboardInput()
        {
            if (Input.GetKeyDown(KeyCode.I) || Input.GetKeyDown(KeyCode.Alpha0))
            {
                ExecuteAction(AgentAction.Wait);
            }
            else if (Input.GetKeyDown(KeyCode.W) || Input.GetKeyDown(KeyCode.Alpha1))
            {
                ExecuteAction(AgentAction.DoNothing);
            }
            else if (Input.GetKeyDown(KeyCode.H) || Input.GetKeyDown(KeyCode.Alpha2))
            {
                ExecuteAction(AgentAction.Look);
            }
            else if (Input.GetKeyDown(KeyCode.L) || Input.GetKeyDown(KeyCode.Alpha3))
            {
                ExecuteAction(AgentAction.Wave);
            }
            else if (Input.GetKeyDown(KeyCode.R) || Input.GetKeyDown(KeyCode.Alpha4))
            {
                ExecuteAction(AgentAction.HandShake);
            }
        }
        
        // Public methods to get current state info
        public string GetCurrentStateDescription()
        {
            switch (CurrentState)
            {
                case PepperState.Idle:
                    return "Idle - Robot is not performing any action";
                case PepperState.Looking:
                    return $"Looking at target for {Mathf.Max(0, lookEndTime - Time.time):F1} more seconds";
                case PepperState.Waving:
                    return "Waving at nearest person";
                case PepperState.Handshaking:
                    return "Performing handshake action";
                case PepperState.PerformingAction:
                    return "Currently performing an action";
                case PepperState.MonitoringNPC:
                    return $"Monitoring NPC: {npcToMonitor?.gameObject.name ?? "None"}";
                default:
                    return "Unknown state";
            }
        }
        
        public string GetCurrentActionDescription()
        {
            if (currentLookTarget != null && isLooking)
            {
                return $"Looking at: {currentLookTarget.name}";
            }
            return "No active action";
        }
    }
}