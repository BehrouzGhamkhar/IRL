using Managers;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Agents.Robot
{
    public class PepperAgent : Agent
    {
        [SerializeField] private PepperController pepperController;
        [SerializeField] private CommunicationManager communicationManager;

        [Header("Episode Settings")]
        [SerializeField, Tooltip("Max number of decisions before timeout")]
        private int maxDecisionSteps = 400;           // ~8–25 seconds depending on Decision Period

        [SerializeField, Tooltip("Max real time per episode (safety)")]
        private float maxEpisodeDurationSeconds = 60f;

        private float episodeStartTime;
        private int decisionStepCount;

        
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
            decisionStepCount = 0;

            if (communicationManager != null)
            {
                communicationManager.ResetSimulation();
            }

            Debug.Log("[PepperAgent] Episode begin – environment reset");
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            if (communicationManager == null)
            {
                sensor.AddObservation(new float[16]); // safety padding
                return;
            }

            // 1. Pepper state (one-hot) – 5 values
            sensor.AddOneHotObservation((int)communicationManager.CurrentPepperState, 5);

            // 2. Current NPC Task - 1 value (task id)
            sensor.AddObservation(communicationManager.CurrentNPCTaskId);                      // single float = task id (0 = no task)

            // 3. Normalized distance – 1 value
            float normDist = Mathf.Clamp01(communicationManager.CurrentDistance / 10f);
            sensor.AddObservation(normDist);

            // 4. Handshake flags
            sensor.AddObservation(communicationManager.IsHandshakeInProgress ? 1f : 0f);
            sensor.AddObservation(communicationManager.CanHandshake ? 1f : 0f);

            // 5. Relative position (x,z) normalized
            if (pepperController && communicationManager.NpcController)
            {
                Vector3 relPos = communicationManager.NpcController.transform.position -
                                 pepperController.transform.position;
                sensor.AddObservation(Mathf.Clamp(relPos.x / 10f, -1f, 1f));
                sensor.AddObservation(Mathf.Clamp(relPos.z / 10f, -1f, 1f));
            }
            else
            {
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
            }

            // Total ~ 5 + 1 + 1 + 2 + 2 = 11 observations 
             // Debug.Log($"[ML-Agents] Collecting {sensor.GetObservationSpec().Shape[0]} observations");

        }

        public override void OnActionReceived(ActionBuffers actionBuffers)
        {
            if (communicationManager == null) return;

            int actionIndex = actionBuffers.DiscreteActions[0]; // 0..4

            PepperController.AgentAction selectedAction = actionIndex switch
            {
                0 => PepperController.AgentAction.DoNothing,
                1 => PepperController.AgentAction.Talk,
                2 => PepperController.AgentAction.Look,
                3 => PepperController.AgentAction.Wave,
                4 => PepperController.AgentAction.HandShake,
                _ => PepperController.AgentAction.DoNothing
            };

            communicationManager.ExecutePepperAction(selectedAction);
            
            // 1. Count this decision
            decisionStepCount++;

            // 2. Small living penalty → encourages finishing quickly
            AddReward(-0.002f);  

            // 3. Check termination conditions
            if (decisionStepCount >= maxDecisionSteps)
            {
                AddReward(-0.2f);
                EndEpisodeWithReason("Timeout: max steps");
                return;
            }

            if (Time.time - episodeStartTime >= maxEpisodeDurationSeconds)
            {
                AddReward(-0.3f);
                EndEpisodeWithReason("Timeout: max duration");
            }
        }
        
        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var discrete = actionsOut.DiscreteActions;

            discrete[0] = 0; // default = DoNothing

            if (Input.GetKey(KeyCode.Alpha1) || Input.GetKey(KeyCode.Keypad1)) discrete[0] = 0; // DoNothing
            if (Input.GetKey(KeyCode.Alpha2) || Input.GetKey(KeyCode.Keypad2)) discrete[0] = 1; // Wait
            if (Input.GetKey(KeyCode.Alpha3) || Input.GetKey(KeyCode.Keypad3)) discrete[0] = 2; // Look
            if (Input.GetKey(KeyCode.Alpha4) || Input.GetKey(KeyCode.Keypad4)) discrete[0] = 3; // Wave
            if (Input.GetKey(KeyCode.Alpha5) || Input.GetKey(KeyCode.Keypad5)) discrete[0] = 4; // HandShake
        }

        // Public methods called from CommunicationManager

        public void EndEpisodeWithReason(string reason)
        {
            Debug.Log($"[Episode End] {reason} | Step: {decisionStepCount} | Time: {(Time.time - episodeStartTime):F1}s | Reward so far: {GetCumulativeReward():F3}");
            EndEpisode();
        }

        public void SetCommunicationManager(CommunicationManager comManager)
        {
            communicationManager = comManager;
        }
    }
}