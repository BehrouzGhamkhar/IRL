using System.Collections.Generic;
using Tasks;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.Events;
using DG.Tweening;

namespace Agents.NPC
{
    public class NPCController : MonoBehaviour
    {
        [Header("Tasks")] public List<NPCTask> allTasks = new List<NPCTask>();

        [Header("Settings")] public float timeBetweenTasks = 3f;

        [Header("State Events")] public UnityEvent<NPCState> onStateChanged;
        public UnityEvent<NPCTask> onTaskChanged;

        // Components
        private NavMeshAgent agent;
        private Animator animator;
        private NPCTask currentTask;
        private Transform currentTarget;
        private NPCState previousState;

        // Timing variables
        private float waitTimer;
        private float taskTimer;
        private bool hasStartedTask = false;
        private bool hasReachedTarget = false;

        public enum NPCState
        {
            Idle,
            WaitingBetweenTasks,
            MovingToTask,
            PerformingTask,
            SearchingForTarget,
            Transitioning
        }

        private NPCState currentState = NPCState.Idle;

        public NPCState CurrentState
        {
            get { return currentState; }
            private set
            {
                if (currentState != value)
                {
                    previousState = currentState;
                    currentState = value;

                    // Reset timing flags when state changes
                    if (currentState == NPCState.MovingToTask)
                    {
                        hasReachedTarget = false;
                    }
                    else if (currentState == NPCState.PerformingTask)
                    {
                        taskTimer = 0f;
                    }
                    else if (currentState == NPCState.WaitingBetweenTasks)
                    {
                        waitTimer = 0f;
                    }
                    else if (currentState == NPCState.Transitioning)
                    {
                        // Reset task flag when transitioning to a new task
                        hasStartedTask = false;
                    }

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

            // Initialize events if they're null
            if (onStateChanged == null)
                onStateChanged = new UnityEvent<NPCState>();
            if (onTaskChanged == null)
                onTaskChanged = new UnityEvent<NPCTask>();

            // Start with waiting between tasks
            CurrentState = NPCState.WaitingBetweenTasks;
            waitTimer = 0f;
        }

        void Update()
        {
            switch (CurrentState)
            {
                case NPCState.WaitingBetweenTasks:
                    UpdateWaitingState();
                    break;

                case NPCState.MovingToTask:
                    UpdateMovingState();
                    break;

                case NPCState.PerformingTask:
                    UpdatePerformingState();
                    break;

                case NPCState.SearchingForTarget:
                    UpdateSearchingState();
                    break;

                case NPCState.Idle:
                    UpdateIdleState();
                    break;

                case NPCState.Transitioning:
                    UpdateTransitioningState();
                    break;
            }
        }

        void UpdateWaitingState()
        {
            waitTimer += Time.deltaTime;

            if (waitTimer >= timeBetweenTasks)
            {
                // Wait time complete, pick a new task
                PickRandomTask();
            }
        }

        void UpdateMovingState()
        {
            if (currentTarget == null)
            {
                CurrentState = NPCState.SearchingForTarget;
                return;
            }

            // Set destination and animation
            agent.SetDestination(currentTarget.position);
            animator.SetBool("IsWalking", true);
            animator.SetBool("IsIdle", false);

            // Check if we've reached the target
            float distance = Vector3.Distance(transform.position, currentTarget.position);
            if (distance <= currentTask.acceptanceRadius && !hasReachedTarget)
            {
                animator.SetBool("IsWalking", false);
                animator.SetBool("IsIdle", true);
                transform.DORotate(currentTarget.eulerAngles, 0.5f, RotateMode.Fast)
                    .OnComplete(() => hasReachedTarget = true);

                // Fire task changed event when we first reach the target
                if (onTaskChanged != null)
                {
                    onTaskChanged.Invoke(currentTask);
                }

                CurrentState = NPCState.PerformingTask;
            }
        }

        void UpdatePerformingState()
        {
            if (!hasStartedTask)
            {
                hasStartedTask = true;
                taskTimer = 0f;

                // Play task-specific animation
                if (!string.IsNullOrEmpty(currentTask.animationName) && animator != null)
                {
                    Debug.Log($"Playing animation clip: {currentTask.animationName}");
                    animator.Play(currentTask.animationName);
                }
                else
                {
                    Debug.Log($"No animation specified for task: {currentTask.taskName}");
                }
            }

            taskTimer += Time.deltaTime;

            if (taskTimer >= currentTask.taskDuration)
            {
                Debug.Log($"Task completed: {currentTask.taskName}");
                hasStartedTask = false;
                currentTask = null;
                currentTarget = null;
                CurrentState = NPCState.WaitingBetweenTasks;
            }
        }

        void UpdateSearchingState()
        {
            if (currentTask == null)
            {
                CurrentState = NPCState.WaitingBetweenTasks;
                return;
            }

            // Try to find the target
            if (string.IsNullOrEmpty(currentTask.targetObjectName))
            {
                // No target specified, just perform the task
                CurrentState = NPCState.PerformingTask;
                return;
            }

            GameObject targetObj = GameObject.Find(currentTask.targetObjectName);
            if (targetObj != null)
            {
                currentTarget = targetObj.transform;
                CurrentState = NPCState.MovingToTask;
            }
            else
            {
                // Target not found, skip this task after a delay
                if (taskTimer > 5f) // 5 second search timeout
                {
                    Debug.LogWarning(
                        $"Target '{currentTask.targetObjectName}' not found for task '{currentTask.taskName}'");
                    currentTask = null;
                    currentTarget = null;
                    CurrentState = NPCState.WaitingBetweenTasks;
                }

                taskTimer += Time.deltaTime;
            }
        }

        void UpdateIdleState()
        {
            // No tasks available, just idle
            if (allTasks.Count > 0)
            {
                CurrentState = NPCState.WaitingBetweenTasks;
            }
        }

        void UpdateTransitioningState()
        {
            // This state is for any transitions or setup
            // Immediately move to next appropriate state
            if (currentTask != null)
            {
                CurrentState = NPCState.SearchingForTarget;
            }
            else
            {
                CurrentState = NPCState.WaitingBetweenTasks;
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
            currentTarget = null;
            hasReachedTarget = false;
            hasStartedTask = false;

            Debug.Log($"NPC now doing: {currentTask.taskName}");

            CurrentState = NPCState.Transitioning;
        }
    }
}