using UnityEngine;
using UnityEngine.AI;
using System.Collections.Generic;
using ScriptableObjects;

public class NPCController : MonoBehaviour
{
    [Header("Tasks")]
    public List<NPCTask> allTasks = new List<NPCTask>();
    
    [Header("Settings")]
    public float timeBetweenTasks = 3f;
    
    // Components
    private NavMeshAgent agent;
    private Animator animator;
    private float timer;
    private NPCTask currentTask;
    private Transform currentTarget;
    
    void Start()
    {
        agent = GetComponent<NavMeshAgent>();
        animator = GetComponentInChildren<Animator>();
        timer = 0;
        // Start first random task
        PickRandomTask();
    }
    
    void Update()
    {
        
        timer += Time.deltaTime;
        if (!currentTask && timer >= timeBetweenTasks)
            PickRandomTask();

        
        // If we have a task, find the target and move there
        if (currentTask)
        {
            if (timer >= timeBetweenTasks + currentTask.taskDuration)
            {
                // Move to next random task
                PickRandomTask();
                timer = 0;
            }
            // Find the target by name if we haven't already
            if (currentTarget == null && !string.IsNullOrEmpty(currentTask.targetObjectName))
            {
                GameObject targetObj = GameObject.Find(currentTask.targetObjectName);
                if (targetObj != null)
                {
                    currentTarget = targetObj.transform;
                }
            }
            
            // Move to target if we found it
            if (currentTarget != null)
            {
                agent.SetDestination(currentTarget.position);
                animator.SetBool("IsWalking", true);
                animator.SetBool("IsIdle", false);
                // Check if close enough
                float distance = Vector3.Distance(transform.position, currentTarget.position);
                if (distance < currentTask.acceptanceRadius)
                {
                    // Play animation if we have one
                    animator.SetBool("IsWalking", false);
                    animator.SetBool("IsIdle", true);
                    if (!string.IsNullOrEmpty(currentTask.animationName) && animator != null)
                    {
                        animator.Play(currentTask.animationName);
                    }
                }
            }
        }
    }
    
    void PickRandomTask()
    {
        if (allTasks.Count == 0) return;
        
        int randomIndex = Random.Range(0, allTasks.Count);
        currentTask = allTasks[randomIndex];
        currentTarget = null; // Reset target so we can find it by name
        
        Debug.Log($"NPC now doing: {currentTask.taskName}");
    }
}