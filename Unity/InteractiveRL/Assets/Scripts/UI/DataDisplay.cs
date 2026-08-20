using Agents;
using Agents.NPC;
using Agents.Robot;
using Managers;
using Managers.Reward;
using TMPro;
using UnityEngine;
using Utilities;

namespace UI
{
    public class DataDisplay : MonoBehaviour
    {
        public CommunicationManager communicationManager;
        public TextMeshProUGUI npcLogText;
        public TextMeshProUGUI pepperLogText;
        public TextMeshProUGUI feedbackText;
        public TextMeshProUGUI rewardText;

        void Start()
        {
            if (communicationManager == null)
                communicationManager = FindFirstObjectByType<CommunicationManager>();

            UpdateDisplay();
        }

        void Update()
        {
            UpdateDisplay();
        }

        private void UpdateDisplay()
        {
            if (!communicationManager) return;

            UpdateNPCDisplay();
            UpdatePepperDisplay();
            UpdateRewardDisplay();
            UpdateFeedbackLog();
        }

        // NPC

        private void UpdateNPCDisplay()
        {
            if (communicationManager.HumanTransform == null || npcLogText == null) return;

            IHumanAgent person = communicationManager.HumanAgent;
            string stateColor = GetNPCStateColor(person.CurrentState);
            string taskText = person.CurrentTask != null ? person.CurrentTask.taskName : "None";

            npcLogText.text =
                $"Person State: <color={stateColor}>{person.CurrentState}</color> \n" +
                $"Task: {taskText} - {communicationManager.CurrentDistance:0.#} ";
        }

        // Pepper

        private void UpdatePepperDisplay()
        {
            if (communicationManager.PepperController == null || pepperLogText == null) return;

            PepperController pepper = communicationManager.PepperController;
            string stateColor = GetPepperStateColor(pepper.CurrentState);

            pepperLogText.text =
                $"Pepper State: <color={stateColor}>{pepper.CurrentState}</color>\n" +
                $"Action: {pepper.CurrentAction}";
        }

        // Cumulative Reward

        private void UpdateRewardDisplay()
        {
            if (communicationManager.PepperAgent == null || rewardText == null) return;

            float reward = communicationManager.PepperAgent.GetCumulativeReward();
            string rewardString = reward > 0 ? "+" + reward : reward.ToString();
            string color = GetFeedbackColor(reward);

            rewardText.text = $"Cumulative Reward:\n<color={color}>{rewardString}</color>";
        }

        // Feedback Log

        private void UpdateFeedbackLog()
        {
            if (feedbackText == null) return;

            var entries = FeedbackLogger.GetEntries();

            if (entries.Length == 0)
            {
                feedbackText.text = "Feedback Log:\n<color=#888888>—</color>";
                return;
            }

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("Feedback Log:");

            foreach (var e in entries)
            {
                string sign = e.value >= 0 ? "+" : "";
                string color = e.value > 0 ? "#00FF00" : (e.value < 0 ? "#FF4444" : "#888888");
                sb.AppendLine($"<color={color}>[{e.source}] {sign}{e.value:0.##}</color>");
            }

            feedbackText.text = sb.ToString().TrimEnd();
        }

        // Color Helpers

        private string GetNPCStateColor(NPCController.NPCState state) => state switch
        {
            NPCController.NPCState.Idle => "#D3D3D3", // gray
            NPCController.NPCState.WaitingBetweenTasks => "#D3D3D3", // gray
            NPCController.NPCState.MovingToTask => "#FFFF00", // yellow
            NPCController.NPCState.PerformingTask => "#00FF00", // green
            NPCController.NPCState.SearchingForTarget => "#FFA500", // orange
            NPCController.NPCState.Transitioning => "#0000FF", // blue
            _ => "#FFFFFF"
        };

        private string GetPepperStateColor(PepperController.PepperState state) => state switch
        {
            PepperController.PepperState.Idle => "#D3D3D3", // gray
            PepperController.PepperState.Looking => "#00FFFF", // cyan
            PepperController.PepperState.Waving => "#FF00FF", // magenta
            PepperController.PepperState.Handshaking => "#0000FF", // blue
            PepperController.PepperState.PerformingAction => "#FFFF00", // yellow
            _ => "#FFFFFF"
        };

        private string GetFeedbackColor(float v) => v switch
        {
            > 0 => "#00FF00",
            < 0 => "#FF4444",
            _ => "#D3D3D3"
        };
    }
}