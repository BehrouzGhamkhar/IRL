using System.Collections;
using System.Collections.Generic;
using Agents;
using Tasks;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.Events;

namespace Agents.NPC
{
    public class NPCController : MonoBehaviour, IHumanAgent
    {
        [Header("Tasks")] public List<NPCTask> allTasks = new();

        [Header("Settings")] [Tooltip("Seconds to wait before picking the next task.")]
        public float timeBetweenTasks = 1f;

        [Header("Reproducibility")]
        [Tooltip("Master seed for this run. Same seed = identical task sequence across runs.")]
        public int masterSeed = -1;

        // This is the actual random generator for this run - never reinitialized
        private System.Random runRandom;

        // Episode counter - tracks how many episodes have occurred
        private int episodeCount;

        // Track tasks selected in current episode
        private int tasksSelectedThisEpisode;

        [Header("References")] [SerializeField]
        private NPCAnimationController animationController;

        [Tooltip("The robot transform — used as the look-at point for random position tasks.")] [SerializeField]
        private Transform robotTransform;

        private GameObject _dummy; // hidden GO used as currentTarget for random/noisy positions

        [Header("State Events")] [SerializeField]
        private UnityEvent<NPCState> onStateChanged = new();

        [SerializeField] private UnityEvent<NPCTask> onTaskChanged = new();

        public UnityEvent<NPCState> OnStateChanged => onStateChanged;
        public UnityEvent<NPCTask> OnTaskChanged => onTaskChanged;

        public NPCTask CurrentTask => currentTask;
        public Transform Transform => transform;

        private NavMeshAgent navAgent;
        private NPCTask currentTask;
        private Transform currentTarget;

        private float waitTimer;
        private float taskTimer;
        private float searchTimer;

        private bool hasStartedTask;
        private bool hasReachedTarget;

        public enum NPCState
        {
            Idle,
            WaitingBetweenTasks,
            Transitioning,
            SearchingForTarget,
            MovingToTask,
            PerformingTask
        }

        private NPCState _currentState = NPCState.Idle;

        public NPCState CurrentState
        {
            get => _currentState;
            private set
            {
                if (_currentState == value) return;
                _currentState = value;

                switch (_currentState)
                {
                    case NPCState.WaitingBetweenTasks:
                        waitTimer = 0f;
                        break;
                    case NPCState.MovingToTask:
                        hasReachedTarget = false;
                        break;
                    case NPCState.PerformingTask:
                        taskTimer = 0f;
                        hasStartedTask = false;
                        break;
                    case NPCState.SearchingForTarget:
                        searchTimer = 0f;
                        break;
                }

                onStateChanged?.Invoke(_currentState);
            }
        }

        NPCState IHumanAgent.CurrentState => CurrentState;

        private void Start()
        {
            navAgent = GetComponent<NavMeshAgent>();
            if (animationController == null)
                animationController = GetComponent<NPCAnimationController>();

            // Initialize with master seed - this stays constant for the entire run
            // InitializeRandom(masterSeed);

            CurrentState = NPCState.WaitingBetweenTasks;
            episodeCount = 0;
            tasksSelectedThisEpisode = 0;
        }

        // Set the master seed from external code (CommunicationManager).
        // Called when the config's seed is passed from Python.
        public void SetMasterSeed(int seed)
        {
            masterSeed = seed;
            runRandom = seed == -1 ? new System.Random() : new System.Random(seed);
            Debug.Log($"[NPC] Random initialized with seed: {seed}");
        }


        // Called at the beginning of each episode to reset episode-specific counters.
        public void OnEpisodeBegin(int episodeNumber)
        {
            episodeCount = episodeNumber;
            tasksSelectedThisEpisode = 0;
        }

        private void FixedUpdate()
        {
            switch (CurrentState)
            {
                case NPCState.WaitingBetweenTasks:
                    UpdateWaiting();
                    break;
                case NPCState.Transitioning:
                    UpdateTransition();
                    break;
                case NPCState.SearchingForTarget:
                    UpdateSearching();
                    break;
                case NPCState.MovingToTask:
                    UpdateMoving();
                    break;
                case NPCState.PerformingTask:
                    UpdatePerforming();
                    break;
                case NPCState.Idle:
                    UpdateIdle();
                    break;
            }
        }

        private void UpdateIdle()
        {
            if (allTasks.Count > 0)
                CurrentState = NPCState.WaitingBetweenTasks;
        }

        private void UpdateWaiting()
        {
            waitTimer += Time.deltaTime;
            if (waitTimer >= timeBetweenTasks)
                PickRandomTask();
        }

        private void UpdateTransition()
        {
            CurrentState = currentTask != null
                ? NPCState.SearchingForTarget
                : NPCState.WaitingBetweenTasks;
        }

        private void UpdateSearching()
        {
            if (currentTask == null)
            {
                CurrentState = NPCState.WaitingBetweenTasks;
                return;
            }

            if (currentTask.randomPosition)
            {
                // Find a random NavMesh point within [minRange, maxRange] of the robot
                Vector3 centre = robotTransform != null ? robotTransform.position : transform.position;
                if (TryGetRandomNavMeshPoint(centre, currentTask.minRange, currentTask.maxRange, out Vector3 point))
                {
                    currentTarget = PlaceDummyFacingRobot(point);
                    CurrentState = NPCState.MovingToTask;
                }
                else
                {
                    searchTimer += Time.deltaTime;
                    if (searchTimer > 5f)
                    {
                        Debug.LogWarning($"[NPC] No NavMesh point found for '{currentTask.taskName}' - skipping.");
                        ClearCurrentTask();
                    }
                }

                return;
            }

            // Fixed target with small position noise
            if (string.IsNullOrEmpty(currentTask.targetObjectName))
            {
                CurrentState = NPCState.PerformingTask;
                return;
            }

            var targetObj = GameObject.Find(currentTask.targetObjectName);
            if (targetObj != null)
            {
                float angle = (float)(runRandom.NextDouble() * System.Math.PI * 2.0);
                float radius = (float)(runRandom.NextDouble() * currentTask.positionNoise);
                Vector3 noise = new Vector3(Mathf.Cos(angle) * radius, 0f, Mathf.Sin(angle) * radius);
                currentTarget = PlaceDummyFacingRobot(targetObj.transform.position + noise);
                CurrentState = NPCState.MovingToTask;
                return;
            }

            searchTimer += Time.deltaTime;
            if (searchTimer > 5f)
            {
                Debug.LogWarning($"[NPC] Target '{currentTask.targetObjectName}' not found - skipping.");
                ClearCurrentTask();
            }
        }

        // Moves (or creates) the dummy GO to position, rotated to face the robot.
        // This gives RotateToTargetCoroutine the correct direction to align the NPC.
        private Transform PlaceDummyFacingRobot(Vector3 position)
        {
            if (_dummy == null)
                _dummy = new GameObject("_NPC_Dummy") { hideFlags = HideFlags.HideInHierarchy };

            _dummy.transform.position = position;

            if (robotTransform != null)
            {
                Vector3 dir = (robotTransform.position - position).normalized;
                if (dir != Vector3.zero)
                    _dummy.transform.rotation = Quaternion.LookRotation(dir);
            }

            return _dummy.transform;
        }

        private bool TryGetRandomNavMeshPoint(Vector3 centre, float minRange, float maxRange, out Vector3 result)
        {
            for (int i = 0; i < 30; i++)
            {
                float angle = (float)(runRandom.NextDouble() * System.Math.PI * 2.0);
                float radius = Mathf.Lerp(minRange, maxRange, (float)runRandom.NextDouble());
                Vector3 candidate = centre + new Vector3(Mathf.Cos(angle) * radius, 0f, Mathf.Sin(angle) * radius);
                if (NavMesh.SamplePosition(candidate, out NavMeshHit hit, 2f, NavMesh.AllAreas))
                {
                    result = hit.position;
                    return true;
                }
            }

            result = Vector3.zero;
            return false;
        }

        private void UpdateMoving()
        {
            if (currentTarget == null)
            {
                CurrentState = NPCState.SearchingForTarget;
                return;
            }

            if (NavMesh.SamplePosition(currentTarget.position, out NavMeshHit hit, 2f, NavMesh.AllAreas))
                navAgent.SetDestination(hit.position);

            animationController?.PlayWalk();

            float dist = Vector3.Distance(transform.position, currentTarget.position);
            if (!hasReachedTarget && dist <= currentTask.acceptanceRadius)
            {
                navAgent.ResetPath();                // stop NavMeshAgent steering so it doesn't drift during task
                navAgent.velocity = Vector3.zero;
                animationController?.PlayIdle();
                StartCoroutine(RotateThenPerform());
            }
        }

        private IEnumerator RotateThenPerform()
        {
            if (animationController != null && currentTarget != null)
                yield return animationController.RotateToTargetCoroutine(currentTarget, 0.5f);

            hasReachedTarget = true;
            CurrentState = NPCState.PerformingTask;
        }

        private void UpdatePerforming()
        {
            if (!hasStartedTask)
            {
                hasStartedTask = true;
                onTaskChanged?.Invoke(currentTask);

                if (currentTask != null && !string.IsNullOrEmpty(currentTask.animationName))
                    animationController?.PlayTaskAnimation(currentTask.animationName);
            }

            taskTimer += Time.deltaTime;

            if (currentTask == null || taskTimer >= currentTask.taskDuration)
            {
                Debug.Log($"[NPC] Task complete: {currentTask?.taskName}");
                ClearCurrentTask();
            }
        }

        // Pick a random task using runRandom generator.
        // This produces a deterministic sequence across episodes for a given seed.
        private void PickRandomTask()
        {
            if (allTasks.Count == 0)
            {
                CurrentState = NPCState.Idle;
                return;
            }

            int taskIndex = runRandom.Next(0, allTasks.Count);
            currentTask = allTasks[taskIndex];
            currentTarget = null;
            hasReachedTarget = false;
            hasStartedTask = false;
            tasksSelectedThisEpisode++;

            Debug.Log(
                $"[NPC] Episode {episodeCount}, Task #{tasksSelectedThisEpisode}: {currentTask.taskName} (index {taskIndex})");
            CurrentState = NPCState.Transitioning;
        }

        /// Force-start a task, skipping the wait timer.
        public void ForceStartTask()
        {
            if (allTasks.Count == 0)
            {
                Debug.LogWarning("[NPC] ForceStartTask: no tasks assigned.");
                CurrentState = NPCState.Idle;
                return;
            }

            int taskIndex = runRandom.Next(0, allTasks.Count);
            currentTask = allTasks[taskIndex];
            currentTarget = null;
            hasReachedTarget = false;
            hasStartedTask = false;
            waitTimer = 0f;
            taskTimer = 0f;
            tasksSelectedThisEpisode++;

            Debug.Log(
                $"[NPC] Episode {episodeCount}, Force-starting task #{tasksSelectedThisEpisode}: {currentTask.taskName}");
            CurrentState = NPCState.Transitioning;
        }

        public void ClearCurrentTask()
        {
            hasStartedTask = false;
            currentTask = null;
            currentTarget = null;
            taskTimer = 0f;
            CurrentState = NPCState.WaitingBetweenTasks;
        }

        public void StopMovement()
        {
            if (navAgent == null) return;
            navAgent.ResetPath();
            navAgent.velocity = Vector3.zero;
        }
    }
}