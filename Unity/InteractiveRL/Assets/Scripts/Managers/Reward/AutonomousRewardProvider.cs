using Agents;
using Agents.NPC;
using Agents.Robot;
using UnityEngine;
using UnityEngine.Events;
using Utilities;

namespace Managers.Reward
{
    // Experiment 1: Robot gets rewards automatically based on task success

    public class AutonomousRewardProvider : MonoBehaviour, IRewardProvider
    {
        [Header("Reward Values")] public float correctReward = 1f;
        public float wrongPenalty = -0.2f;

        [Header("Distance Settings")] public float interactionDistance = 5f;

        public bool IsEnabled { get; set; } = true;
        public UnityEvent<float> OnReward { get; } = new UnityEvent<float>();

        private int lastTaskId = -1;

        // Called by CommunicationManager when robot takes an action
        public float CheckReward(int taskId, PepperController.AgentAction action, float distance)
        {
            if (!IsEnabled) return 0;

            bool isRetry = (taskId == lastTaskId);
            lastTaskId = taskId;

            float reward = CalculateReward(taskId, action, distance);
            // Slightly penalise retries so the model doesn't learn wrong-then-right as a strategy
            if (isRetry && reward > 0)
                reward *= 0.5f;

            Debug.Log($"[AutoReward] {reward} for task {taskId} action {action}");

            FeedbackLogger.Add("Auto", reward); // ← log
            OnReward?.Invoke(reward);
            return reward;
        }

        private float CalculateReward(int taskId, PepperController.AgentAction action, float distance)
        {
            // Task ID mapping: 2 = Handshake, 7 = Wave, 6 = Talk
            switch (taskId)
            {
                case 2: return action == PepperController.AgentAction.HandShake ? correctReward : wrongPenalty;
                case 7: return action == PepperController.AgentAction.Wave ? correctReward : wrongPenalty;
                case 6: return action == PepperController.AgentAction.Talk ? correctReward : wrongPenalty;
                default:
                    if (distance <= interactionDistance)
                        return action == PepperController.AgentAction.Look ? correctReward : wrongPenalty;
                    else
                        return action == PepperController.AgentAction.DoNothing ? 0.001f : wrongPenalty;
            }
        }

        public void Reset()
        {
            lastTaskId = -1;
        }
    }
}