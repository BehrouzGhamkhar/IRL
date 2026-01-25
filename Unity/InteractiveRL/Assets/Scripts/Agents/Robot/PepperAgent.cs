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

        private void Start()
        {
            if (pepperController == null)
                pepperController = GetComponent<PepperController>();
            if (communicationManager == null)
                communicationManager = FindFirstObjectByType<CommunicationManager>();
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // 1. Pepper state (one-hot style)
            sensor.AddOneHotObservation((int)communicationManager.CurrentPepperState, 5); // 5 possible states

            // 2. NPC state
            sensor.AddOneHotObservation((int)communicationManager.CurrentNPCState, 6);

            // 3. Normalized distance
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
                sensor.AddObservation(relPos.x / 10f);
                sensor.AddObservation(relPos.z / 10f);
            }
            else
            {
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
            }

            // Total ~ 5 + 6 + 1 + 2 + 2 = 16 observations 
        }

        public override void OnActionReceived(ActionBuffers actionBuffers)
        {
            int actionIndex = actionBuffers.DiscreteActions[0]; // 0..4

            PepperController.AgentAction action = actionIndex switch
            {
                0 => PepperController.AgentAction.DoNothing,
                1 => PepperController.AgentAction.Wait,
                2 => PepperController.AgentAction.Look,
                3 => PepperController.AgentAction.Wave,
                4 => PepperController.AgentAction.HandShake,
                _ => PepperController.AgentAction.DoNothing
            };

            if (communicationManager != null)
                communicationManager.ExecutePepperAction(action);
            else if (pepperController != null)
                pepperController.ExecuteAction(action);
        }

        public override void OnEpisodeBegin()
        {
            if (communicationManager != null)
            {
                communicationManager.ResetHandshake();
            }
        }


        public void SetCommunicationManager(CommunicationManager comManager)
        {
            this.communicationManager = comManager;
        }
    }
}