using Agents.NPC;
using Agents.Robot;
using Managers;
using TMPro;
using UnityEngine;

public class DataDisplay : MonoBehaviour
{
    public CommunicationManager communicationManager;
    public TextMeshProUGUI npcLogText;
    public TextMeshProUGUI pepperLogText;
    
    void Start()
    {
        // Get references if not set in inspector
        if (communicationManager == null)
        {
            communicationManager = FindFirstObjectByType<CommunicationManager>();
        }
        
        // Display initial state
        UpdateDisplay();
    }
    
    void Update()
    {
        // Update display every frame
        UpdateDisplay();
    }
    
    private void UpdateDisplay()
    {
        // Update NPC Display
        if (communicationManager != null && communicationManager.NpcController != null && npcLogText != null)
        {
            NPCController npc = communicationManager.NpcController;
            string stateText = npc.CurrentState.ToString();
            string taskText = npc.CurrentTask != null ? npc.CurrentTask.taskName : "None";
            
            // Color code the state
            string stateColor = GetNPCStateColor(npc.CurrentState);

            // Apply color using rich text tags
            npcLogText.text = $"NPC State: <color={stateColor}>{stateText}</color>\n" +
                              $"Task: {taskText}";
        }
        
        // Update Pepper Display
        if (communicationManager != null && communicationManager.PepperController != null && pepperLogText != null)
        {
            PepperController pepper = communicationManager.PepperController;
            string stateText = pepper.CurrentState.ToString();
            
            // Color code the state
            string stateColor = GetPepperStateColor(pepper.CurrentState);

            // Apply color using rich text tags
            pepperLogText.text = $"Pepper State: <color={stateColor}>{stateText}</color>";
        }
    }
    
    private string GetNPCStateColor(NPCController.NPCState state)
    {
        switch (state)
        {
            case NPCController.NPCState.Idle:
                return "#D3D3D3"; // gray
            case NPCController.NPCState.WaitingBetweenTasks:
                return "#D3D3D3"; // gray
            case NPCController.NPCState.MovingToTask:
                return "#FFFF00"; // yellow
            case NPCController.NPCState.PerformingTask:
                return "#00FF00"; // green
            case NPCController.NPCState.SearchingForTarget:
                return "#FFA500"; // orange
            case NPCController.NPCState.Transitioning:
                return "#0000FF"; // blue
            default:
                return "#FFFFFF"; // white
        }
    }

    private string GetPepperStateColor(PepperController.PepperState state)
    {
        switch (state)
        {
            case PepperController.PepperState.Idle:
                return "#D3D3D3"; // gray
            case PepperController.PepperState.Looking:
                return "#00FFFF"; // cyan
            case PepperController.PepperState.Waving:
                return "#FF00FF"; // magenta
            case PepperController.PepperState.Handshaking:
                return "#0000FF"; // blue
            case PepperController.PepperState.PerformingAction:
                return "#FFFF00"; // yellow
            default:
                return "#FFFFFF"; // white
        }
    }
}