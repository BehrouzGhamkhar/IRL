using Agents.NPC;
using Agents.Robot;
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
            
            npcLogText.text = $"<color={stateColor}><b>NPC State: {stateText}</b></color>\n" +
                            $"Task: {taskText}";
        }
        
        // Update Pepper Display
        if (communicationManager != null && communicationManager.PepperController != null && pepperLogText != null)
        {
            PepperController pepper = communicationManager.PepperController;
            string stateText = pepper.CurrentState.ToString();
            
            // Color code the state
            string stateColor = GetPepperStateColor(pepper.CurrentState);
            
            pepperLogText.text = $"<color={stateColor}><b>Pepper State: {stateText}</b></color>";
        }
    }
    
    private string GetNPCStateColor(NPCController.NPCState state)
    {
        switch (state)
        {
            case NPCController.NPCState.Idle:
                return "gray";
            case NPCController.NPCState.WaitingBetweenTasks:
                return "lightgray";
            case NPCController.NPCState.MovingToTask:
                return "yellow";
            case NPCController.NPCState.PerformingTask:
                return "green";
            case NPCController.NPCState.SearchingForTarget:
                return "orange";
            case NPCController.NPCState.Transitioning:
                return "blue";
            default:
                return "white";
        }
    }
    
    private string GetPepperStateColor(PepperController.PepperState state)
    {
        switch (state)
        {
            case PepperController.PepperState.Idle:
                return "gray";
            case PepperController.PepperState.Looking:
                return "cyan";
            case PepperController.PepperState.Waving:
                return "magenta";
            case PepperController.PepperState.Handshaking:
                return "blue";
            case PepperController.PepperState.PerformingAction:
                return "yellow";
            default:
                return "white";
        }
    }
}