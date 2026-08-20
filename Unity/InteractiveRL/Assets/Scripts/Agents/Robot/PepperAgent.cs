using Managers;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Agents.Robot
{
    public class PepperAgent : Agent
    {
        // Inspector 
        [Header("References")] [SerializeField]
        private PepperController pepperController;

        [SerializeField] private CommunicationManager communicationManager;

        [Header("Episode Limits")] [SerializeField]
        private int maxDecisionSteps = 2000;

        [SerializeField] private float maxEpisodeDurationSeconds = 60f;

        private float episodeStartTime;
        private int stepCount;

        private void Start()
        {
            if (pepperController == null)
                pepperController = GetComponent<PepperController>();

            if (communicationManager == null)
                communicationManager = FindFirstObjectByType<CommunicationManager>();
        }

        public override void OnEpisodeBegin()
        {
            episodeStartTime = Time.time;
            stepCount = 0;

            // Let CommunicationManager know a new episode is starting
            communicationManager?.OnEpisodeBegin();
            communicationManager?.ResetSimulation();
            communicationManager?.ResetRewardGate();

            Debug.Log($"[PepperAgent] Episode {communicationManager?.GetCurrentEpisodeNumber()} started.");
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            if (communicationManager == null)
            {
                sensor.AddObservation(new float[12]); 
                return;
            }

            // [0-4] Pepper state (one-hot, 5 values)
            sensor.AddOneHotObservation((int)communicationManager.CurrentPepperState, 5);
            
            // sensor.AddOneHotObservation(communicationManager.CurrentNPCTaskId, 10);

            // [5] Normalised distance to human
            sensor.AddObservation(Mathf.Clamp01(communicationManager.CurrentDistance / 10f));

            // [6] Handshake in progress
            sensor.AddObservation(communicationManager.IsHandshakeInProgress ? 1f : 0f);

            // [7] Handshake available
            sensor.AddObservation(communicationManager.CanHandshake ? 1f : 0f);

            // ── Body-signal observations (replace the old 8-value task one-hot) ──

            // [8]  Wrist height above floor, normalised [0, 1]
            //       Low  (0.0) -> hand at hip / side
            //       High (1.0) -> hand raised above head
             sensor.AddObservation(communicationManager.WristHeight);

            // [9]  Wrist-to-core distance, normalised [0, 1]
            //       Low  (0.0) -> arm folded / crossed
            //       High (1.0) -> arm fully extended away from body
             sensor.AddObservation(communicationManager.WristToCoreDistance);

            // [10] Body orientation relative to Pepper, normalised [-1, 1]
            //       +1 -> human facing directly towards Pepper
            //       -1 -> human facing directly away from Pepper
             sensor.AddObservation(communicationManager.BodyOrientation);

            // [11] Gaze direction relative to Pepper, normalised [-1, 1]
            //       +1 -> human looking directly at Pepper
            //       -1 -> human looking directly away from Pepper
             sensor.AddObservation(communicationManager.GazeDirection);

            
            // Total: 5 (PepperState) + 1 (distance) + 2 (handshake) + 4 (body data) = 12
        }

        public override void OnActionReceived(ActionBuffers actionBuffers)
        {
            if (communicationManager == null) return;

            var action = (PepperController.AgentAction)actionBuffers.DiscreteActions[0];
            communicationManager.ExecutePepperAction(action);

            stepCount++;

            // Episode termination checks
            if (stepCount >= maxDecisionSteps)
                EndEpisodeWithReason("Max steps reached");
            else if (Time.time - episodeStartTime >= maxEpisodeDurationSeconds)
                EndEpisodeWithReason("Max duration reached");
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            // Keyboard override for manual testing
            var d = actionsOut.DiscreteActions;
            d[0] = 0; // default: DoNothing

            if (Input.GetKey(KeyCode.Alpha1) || Input.GetKey(KeyCode.Keypad1)) d[0] = 0;
            if (Input.GetKey(KeyCode.Alpha2) || Input.GetKey(KeyCode.Keypad2)) d[0] = 1;
            if (Input.GetKey(KeyCode.Alpha3) || Input.GetKey(KeyCode.Keypad3)) d[0] = 2;
            if (Input.GetKey(KeyCode.Alpha4) || Input.GetKey(KeyCode.Keypad4)) d[0] = 3;
            if (Input.GetKey(KeyCode.Alpha5) || Input.GetKey(KeyCode.Keypad5)) d[0] = 4;
        }
        
        public void EndEpisodeWithReason(string reason)
        {
            Debug.Log($"[PepperAgent] Episode ended - {reason} | " +
                      $"Steps: {stepCount} | " +
                      $"Time: {(Time.time - episodeStartTime):F1}s | " +
                      $"Total reward: {GetCumulativeReward():F3}");
            EndEpisode();
        }

        public void SetCommunicationManager(CommunicationManager manager)
        {
            communicationManager = manager;
        }
    }
}