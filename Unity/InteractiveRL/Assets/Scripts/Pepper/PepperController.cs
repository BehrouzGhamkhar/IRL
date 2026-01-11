using UnityEngine;
using System.Collections;
using System;
using DG.Tweening;
using Pepper;
using ScriptableObjects;

public class PepperController : MonoBehaviour
{
    [SerializeField] private PepperAnimationController animationController;
    [SerializeField] private Transform headBone;
    [SerializeField] private float headRotationSpeed = 5f;
    [SerializeField] private float lookAtDuration = 3f;
    
    [Header("NPC Monitoring")]
    [SerializeField] private NPCController npcToMonitor; // Drag ONE NPC here in the inspector
    
    private Transform currentLookTarget;
    private float lookEndTime;
    private bool isLooking;

    public enum AgentAction
    {
        DoNothing = 0,
        Wait = 1,
        Look = 2,
        Wave = 3,
        HandShake = 6
    };

    void Start()
    {
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
    }
    
    private void OnNPCTaskChanged(NPCTask newTask)
    {
        LogNPCTask(newTask);
    }
    
    private void LogNPCState(NPCController.NPCState state)
    {
        string npcName = npcToMonitor != null ? npcToMonitor.gameObject.name : "Unknown NPC";
        // todo: I can use the npc state data here
    }
    
    private void LogNPCTask(NPCTask task)
    {
        string npcName = npcToMonitor != null ? npcToMonitor.gameObject.name : "Unknown NPC";
        
        if (task != null)
        {
            // todo: I can use the npc task data here
        }
        else
        {
            Debug.Log($"[NPC Task] '{npcName}' has no current task");
        }
    }
    
    #endregion

    private void ExecuteAction(AgentAction rAction)
    {
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
                break;
                
            default:
                Debug.LogWarning($"Unhandled action: {rAction}");
                break;
        }
    }

    #region Action Implementations
    
    private void ActionWait()
    {
        animationController.PlayIdle();
    }

    private void ActionLook()
    {
        var closestPerson = FindNearestPerson();
        currentLookTarget = closestPerson.transform.Find("HeadPosition");

        if (currentLookTarget != null)
        {
            isLooking = true;
            lookEndTime = Time.time + lookAtDuration;
        }
    }

    private void ActionWave()
    {
        animationController.PlayWave();
    }

    IEnumerator ActionHandshake(float delayTime)
    {
        animationController.PlayTryHandshake();
        var closestPerson = FindNearestPerson();
        yield return new WaitForSeconds(delayTime);
        if (closestPerson != null)
        {
            Vector3 targetPosition = closestPerson.position;
            if (Vector3.Distance(transform.position, targetPosition) < 2.0f)
            {
                animationController.PlayHandshake();
            }
            else
            {
                Debug.LogWarning("Too far to handshake.");
                animationController.PlayIdle();
            }
        }
        else
        {
            Debug.LogWarning("No person found to handshake with.");
            animationController.PlayIdle();
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
            return closestPerson;
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
}