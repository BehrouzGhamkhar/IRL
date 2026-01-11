using Agents.NPC;
using Agents.Robot;
using Tasks;
using TMPro;
using UnityEngine;

public class DataDisplay : MonoBehaviour
{
    public NPCController npcController;
    public PepperController pepperController;
    public TextMeshProUGUI npcLogText;
    public TextMeshProUGUI pepperLogText;
    
    void Start()
    {
        // Subscribe to NPC events
        if (npcController != null)
        {
            // Subscribe to state changes
            if (npcController.onStateChanged != null)
            {
                npcController.onStateChanged.AddListener(OnNPCStateChanged);
            }
            
            // Subscribe to task changes
            if (npcController.onTaskChanged != null)
            {
                npcController.onTaskChanged.AddListener(OnNPCTaskChanged);
            }
            
            // Display initial state
            UpdateDisplay();
        }
    }
    
    void OnDestroy()
    {
        // Clean up event listeners
        if (npcController != null)
        {
            if (npcController.onStateChanged != null)
            {
                npcController.onStateChanged.RemoveListener(OnNPCStateChanged);
            }
            
            if (npcController.onTaskChanged != null)
            {
                npcController.onTaskChanged.RemoveListener(OnNPCTaskChanged);
            }
        }
    }
    
    // Event handler for NPC state changes
    private void OnNPCStateChanged(NPCController.NPCState newState)
    {
        Debug.Log($"<color={GetStateColor(newState)}>[NPC State] State changed to: <b>{newState}</b></color>");
        UpdateDisplay();
    }
    
    // Event handler for NPC task changes
    private void OnNPCTaskChanged(NPCTask newTask)
    {
        if (newTask != null)
        {
            Debug.Log($"[NPC Task] Started task: <b>{newTask.taskName}</b> " +
                     $"(Target: {newTask.targetObjectName})");
        }
        else
        {
            Debug.Log($"[NPC Task] Task cleared");
        }
        UpdateDisplay();
    }
    
    // Update the display text
    private void UpdateDisplay()
    {
        if (npcController != null && npcLogText != null)
        {
            string stateText = npcController.CurrentState.ToString();
            string taskText = npcController.CurrentTask != null 
                ? npcController.CurrentTask.taskName 
                : "None";
                
            string color = GetStateColor(npcController.CurrentState);
            npcLogText.text = $"<color={color}>NPC State: {stateText}</color>\n" +
                            $"Task: {taskText}";
        }
    }
    
    private string GetStateColor(NPCController.NPCState state)
    {
        switch (state)
        {
            case NPCController.NPCState.Idle:
                return "gray";
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
}