using System.Collections;
using UnityEngine;
using UnityEngine.Events;

namespace Agents.Robot
{
    public class PepperController : MonoBehaviour
    {
        [Header("References")] [SerializeField]
        private PepperAnimationController animationController;

        [SerializeField] private Transform headBone;

        [Header("Look Settings")] [SerializeField]
        private float headRotationSpeed = 5f;

        [SerializeField] private float lookAtDuration = 3f;

        [Header("Handshake Settings")] [SerializeField]
        private float handshakeAttemptDelay = 2f;

        [SerializeField] private float handshakeReachDistance = 2f;

        [Header("State Events")] public UnityEvent<PepperState> onStateChanged;
        public UnityEvent<AgentAction> onActionPerformed;

        public enum AgentAction
        {
            DoNothing = 0,
            Talk = 1,
            Look = 2,
            Wave = 3,
            HandShake = 4
        }

        public enum PepperState
        {
            Idle,
            Looking,
            Waving,
            Handshaking,
            PerformingAction
        }

        private PepperState currentState = PepperState.Idle;

        public PepperState CurrentState
        {
            get => currentState;
            set
            {
                if (currentState == value) return;
                currentState = value;
                onStateChanged?.Invoke(currentState);
            }
        }

        public AgentAction CurrentAction { get; private set; }
        private Transform currentLookTarget;
        private float lookEndTime;
        private bool isLooking;

        private void Start()
        {
            onStateChanged ??= new UnityEvent<PepperState>();
            onActionPerformed ??= new UnityEvent<AgentAction>();

            if (animationController == null)
                animationController = GetComponent<PepperAnimationController>();

            if (animationController == null)
                Debug.LogError("[Pepper] PepperAnimationController not found!");

            CurrentState = PepperState.Idle;
        }

        private void Update()
        {
            UpdateHeadLook();
        }

        private void UpdateHeadLook()
        {
            if (!isLooking) return;

            if (Time.time < lookEndTime && currentLookTarget != null)
            {
                CurrentState = PepperState.Looking;

                var direction = currentLookTarget.position - headBone.position;
                var targetRotation = Quaternion.LookRotation(direction);
                headBone.rotation = Quaternion.Slerp(
                    headBone.rotation, targetRotation,
                    Time.deltaTime * headRotationSpeed);
            }
            else
            {
                StopLooking();
            }
        }

        //Execute a single agent action. Called by CommunicationManager.
        public void ExecuteAction(AgentAction action)
        {
            CurrentAction = action;
            CurrentState = PepperState.PerformingAction;
            onActionPerformed?.Invoke(action);

            switch (action)
            {
                case AgentAction.Talk:
                    // DoLook(); // Pepper looks at the person while talking
                    DoTalk();
                    break;

                case AgentAction.Look:
                    DoLook();
                    break;

                case AgentAction.Wave:
                    DoLook();
                    DoWave();
                    break;

                case AgentAction.HandShake:
                    // DoLook();
                    StartCoroutine(DoHandshake());
                    break;

                case AgentAction.DoNothing:
                    CurrentState = PepperState.Idle;
                    break;

                default:
                    Debug.LogWarning($"[Pepper] Unhandled action: {action}");
                    CurrentState = PepperState.Idle;
                    break;
            }
        }

        public void StopLooking()
        {
            isLooking = false;
            if (CurrentState == PepperState.Looking)
                CurrentState = PepperState.Idle;
        }

        private void DoTalk()
        {
            animationController?.PlayIdle();
            Debug.Log("[Pepper] Talking");
            CurrentState = PepperState.Idle;
        }

        private void DoLook()
        {
            var target = FindNearestPersonHead();
            if (target != null)
            {
                currentLookTarget = target;
                isLooking = true;
                lookEndTime = Time.time + lookAtDuration;
            }
            else
            {
                CurrentState = PepperState.Idle;
            }
        }

        private void DoWave()
        {
            animationController?.PlayWave();
            Debug.Log("[Pepper] Waving");
            CurrentState = PepperState.Waving;
            StartCoroutine(ResetAfterAnimation(PepperState.Waving));
        }

        private IEnumerator DoHandshake()
        {
            animationController?.PlayTryHandshake();
            Debug.Log("[Pepper] Attempting handshake…");
            CurrentState = PepperState.Handshaking;

            var nearestPerson = FindNearestPerson();
            yield return new WaitForSeconds(handshakeAttemptDelay);

            if (nearestPerson != null &&
                Vector3.Distance(transform.position, nearestPerson.position) < handshakeReachDistance)
            {
                animationController?.PlayHandshake();
                Debug.Log("[Pepper] Handshake successful!");
            }
            else
            {
                Debug.LogWarning("[Pepper] Too far to complete handshake.");
                animationController?.PlayIdle();
                CurrentState = PepperState.Idle;
            }

            StartCoroutine(ResetAfterAnimation(PepperState.Handshaking));
        }

        private IEnumerator ResetAfterAnimation(PepperState stateToReset)
        {
            yield return new WaitForSeconds(1.5f);
            if (CurrentState == stateToReset)
                CurrentState = PepperState.Idle;
        }

        //Returns the "HeadPosition" child of the nearest tagged Person.
        private Transform FindNearestPersonHead()
        {
            var person = FindNearestPerson();
            return person != null ? person.Find("HeadPosition") : null;
        }

        private Transform FindNearestPerson()
        {
            var people = GameObject.FindGameObjectsWithTag("Person");
            if (people.Length == 0)
            {
                Debug.LogWarning("[Pepper] No GameObjects tagged 'Person' found.");
                return null;
            }

            Transform closest = null;
            float minDist = float.MaxValue;

            foreach (var p in people)
            {
                float d = Vector3.Distance(transform.position, p.transform.position);
                if (d < minDist)
                {
                    minDist = d;
                    closest = p.transform;
                }
            }

            return closest;
        }

        public string StateDescription() => CurrentState switch
        {
            PepperState.Idle => "Idle",
            PepperState.Looking => "Looking at target",
            PepperState.Waving => "Waving",
            PepperState.Handshaking => "Handshaking",
            PepperState.PerformingAction => "Performing action",
            _ => "Unknown"
        };
    }
}