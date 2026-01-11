using UnityEngine;
using UnityEngine.AI;
using System.Collections.Generic;
using ScriptableObjects;
using UnityEngine.Events; // Add this namespace

public class NPCController : MonoBehaviour
{
    [Header("Tasks")]
    public List<NPCTask> allTasks = new List<NPCTask>();
    
    [Header("Settings")]
    public float timeBetweenTasks = 3f;
    
    [Header("State Events")]
    public UnityEvent<NPCState> onStateChanged; // Event that passes the new state
    public UnityEvent<NPCTask> onTaskChanged; // Event for task changes
    
    // Components
    private NavMeshAgent agent;
    private Animator animator;
    private float timer;
    private NPCTask currentTask;
    private Transform currentTarget;
    private NPCState previousState; // Track previous state to avoid duplicate events
    
    public enum NPCState
    {
        Idle,
        MovingToTask,
        PerformingTask,
        SearchingForTarget,
        Transitioning
    }
    
    private NPCState currentState = NPCState.Idle;
    
    // Modified property to fire event when state changes
    public NPCState CurrentState 
    { 
        get { return currentState; }
        private set
        {
            if (currentState != value)
            {
                previousState = currentState;
                currentState = value;
                // Fire the event
                if (onStateChanged != null)
                {
                    onStateChanged.Invoke(currentState);
                }
            }
        }
    }
    
    public NPCTask CurrentTask => currentTask;

    void Start()
    {
        agent = GetComponent<NavMeshAgent>();
        animator = GetComponentInChildren<Animator>();
        timer = 0;
        
        // Initialize events if they're null
        if (onStateChanged == null)
            onStateChanged = new UnityEvent<NPCState>();
        if (onTaskChanged == null)
            onTaskChanged = new UnityEvent<NPCTask>();
        
        // Start first random task
        PickRandomTask();
        CurrentState = NPCState.Idle;
    }
    
    void Update()
    {
        timer += Time.deltaTime;
        
        if (!currentTask && timer >= timeBetweenTasks)
        {
            PickRandomTask();
            CurrentState = NPCState.Transitioning;
        }
        
        // If we have a task, find the target and move there
        if (currentTask)
        {
            if (timer >= timeBetweenTasks + currentTask.taskDuration)
            {
                // Move to next random task
                PickRandomTask();
                timer = 0;
                CurrentState = NPCState.Transitioning;
            }
            
            // Find the target by name if we haven't already
            if (currentTarget == null && !string.IsNullOrEmpty(currentTask.targetObjectName))
            {
                GameObject targetObj = GameObject.Find(currentTask.targetObjectName);
                if (targetObj != null)
                {
                    currentTarget = targetObj.transform;
                    CurrentState = NPCState.SearchingForTarget;
                }
            }
            
            // Move to target if we found it
            if (currentTarget != null)
            {
                agent.SetDestination(currentTarget.position);
                animator.SetBool("IsWalking", true);
                animator.SetBool("IsIdle", false);
                CurrentState = NPCState.MovingToTask;
                
                // Check if close enough
                float distance = Vector3.Distance(transform.position, currentTarget.position);
                if (distance < currentTask.acceptanceRadius)
                {
                    // Play animation if we have one
                    animator.SetBool("IsWalking", false);
                    animator.SetBool("IsIdle", true);
                    CurrentState = NPCState.PerformingTask;
                    
                    // Fire task changed event
                    if (onTaskChanged != null)
                    {
                        onTaskChanged.Invoke(currentTask);
                    }

                    if (!string.IsNullOrEmpty(currentTask.animationName) && animator != null)
                    {
                        animator.Play(currentTask.animationName);
                    }
                }
            }
            else if (CurrentState != NPCState.SearchingForTarget)
            {
                CurrentState = NPCState.SearchingForTarget;
            }
        }
        else if (CurrentState != NPCState.Idle && CurrentState != NPCState.Transitioning)
        {
            CurrentState = NPCState.Idle;
        }
    }
    
    void PickRandomTask()
    {
        if (allTasks.Count == 0) 
        {
            CurrentState = NPCState.Idle;
            return;
        }
        
        int randomIndex = Random.Range(0, allTasks.Count);
        currentTask = allTasks[randomIndex];
        currentTarget = null; // Reset target so we can find it by name
        
        CurrentState = NPCState.Transitioning;
        
        Debug.Log($"NPC now doing: {currentTask.taskName}");
    }
}