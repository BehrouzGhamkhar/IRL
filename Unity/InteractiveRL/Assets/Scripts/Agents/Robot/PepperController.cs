using System.Collections;
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

        [Header("State Events")] public UnityEvent<PepperState> onStateChanged;
        public UnityEvent<AgentAction> onActionPerformed;

        private Transform currentLookTarget;
        private float lookEndTime;
        private bool isLooking;
        private PepperState currentState = PepperState.Idle;
        private AgentAction currentAction = AgentAction.DoNothing;

        public AgentAction CurrentAction
        {
            get => currentAction;
            set => currentAction = value;
        }

        private PepperState previousState;

        public enum AgentAction
        {
            DoNothing = 0,
            Talk = 1,
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
            PerformingAction
        }

        public PepperState CurrentState
        {
            get { return currentState; }
            set
            {
                if (currentState != value)
                {
                    previousState = currentState;
                    currentState = value;
                    onStateChanged?.Invoke(currentState);
                }
            }
        }


        void Start()
        {
            if (onStateChanged == null)
                onStateChanged = new UnityEvent<PepperState>();
            if (onActionPerformed == null)
                onActionPerformed = new UnityEvent<AgentAction>();

            if (animationController == null)
            {
                animationController = GetComponent<PepperAnimationController>();
                if (animationController == null)
                    Debug.LogError("Robot Animator not found!");
            }

            CurrentState = PepperState.Idle;
        }

        void Update()
        {
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
                StopLooking();
            }
        }

        public void ExecuteAction(AgentAction rAction)
        {
            CurrentState = PepperState.PerformingAction;
            currentAction = rAction;
            onActionPerformed?.Invoke(rAction);

            switch (rAction)
            {
                case AgentAction.Talk:
                    ActionTalk();
                    ActionLook();
                    break;

                case AgentAction.Look:
                    ActionLook();
                    break;

                case AgentAction.Wave:
                    ActionLook();
                    ActionWave();
                    break;

                case AgentAction.HandShake:
                    float tryHandShakeTime = 2.0f;
                    ActionLook();
                    StartCoroutine(ActionHandshake(tryHandShakeTime));
                    break;

                case AgentAction.DoNothing:
                    CurrentState = PepperState.Idle;
                    break;

                default:
                    Debug.LogWarning($"Unhandled action: {rAction}");
                    CurrentState = PepperState.Idle;
                    break;
            }
        }

        #region Action Implementations

        private void ActionTalk()
        {
            animationController.PlayIdle();
            Debug.Log("[Pepper Action] Talking");
            CurrentState = PepperState.Idle;
        }

        private void ActionLook()
        {
            currentLookTarget = FindNearestPerson()?.Find("HeadPosition");

            if (currentLookTarget != null)
            {
                isLooking = true;
                lookEndTime = Time.time + lookAtDuration;
                //Debug.Log("[Pepper Action] Looking at nearest person");
            }
            else
            {
                //Debug.LogWarning("[Pepper Action] No person found to look at");
                CurrentState = PepperState.Idle;
            }
        }

        public void StopLooking()
        {
            isLooking = false;
            if (CurrentState == PepperState.Looking)
                CurrentState = PepperState.Idle;
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

            if (closestPerson != null &&
                Vector3.Distance(transform.position, closestPerson.position) < 2.0f)
            {
                animationController.PlayHandshake();
                Debug.Log("[Pepper Action] Handshake successful");
            }
            else
            {
                Debug.LogWarning("[Pepper Action] Too far to handshake");
                animationController.PlayIdle();
                CurrentState = PepperState.Idle;
            }

            StartCoroutine(ResetStateAfterAnimation(PepperState.Handshaking));
        }

        IEnumerator ResetStateAfterAnimation(PepperState stateToReset)
        {
            yield return new WaitForSeconds(1.5f);
            if (CurrentState == stateToReset)
                CurrentState = PepperState.Idle;
        }

        #endregion

        private Transform FindNearestPerson()
        {
            GameObject[] people = GameObject.FindGameObjectsWithTag("Person");
            float closestDistance = float.MaxValue;
            Transform closestPerson = null;

            foreach (var person in people)
            {
                float distance = Vector3.Distance(transform.position, person.transform.position);
                if (distance < closestDistance)
                {
                    closestDistance = distance;
                    closestPerson = person.transform;
                }
            }

            if (closestPerson == null)
                Debug.LogWarning("No person found to look at.");

            return closestPerson;
        }

        public string GetCurrentStateDescription()
        {
            switch (CurrentState)
            {
                case PepperState.Idle: return "Idle";
                case PepperState.Looking: return $"Looking at target";
                case PepperState.Waving: return "Waving";
                case PepperState.Handshaking: return "Handshaking";
                case PepperState.PerformingAction: return "Performing action";
                default: return "Unknown state";
            }
        }
    }
}