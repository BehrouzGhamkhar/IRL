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
            if (npcController.onStateChanged != null)
            {
                npcController.onStateChanged.AddListener(OnNPCStateChanged);
            }
            
            if (npcController.onTaskChanged != null)
            {
                npcController.onTaskChanged.AddListener(OnNPCTaskChanged);
            }
        }
        
        // Subscribe to Pepper events
        if (pepperController != null)
        {
            if (pepperController.onStateChanged != null)
            {
                pepperController.onStateChanged.AddListener(OnPepperStateChanged);
            }
            
            if (pepperController.onActionPerformed != null)
            {
                pepperController.onActionPerformed.AddListener(OnPepperActionPerformed);
            }
            
            if (pepperController.onNPCStateChanged != null)
            {
                pepperController.onNPCStateChanged.AddListener(OnPepperNPCStateChanged);
            }
            
            if (pepperController.onNPCTaskChanged != null)
            {
                pepperController.onNPCTaskChanged.AddListener(OnPepperNPCTaskChanged);
            }
        }
        
        // Display initial state
        UpdateDisplay();
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
        
        if (pepperController != null)
        {
            if (pepperController.onStateChanged != null)
            {
                pepperController.onStateChanged.RemoveListener(OnPepperStateChanged);
            }
            
            if (pepperController.onActionPerformed != null)
            {
                pepperController.onActionPerformed.RemoveListener(OnPepperActionPerformed);
            }
            
            if (pepperController.onNPCStateChanged != null)
            {
                pepperController.onNPCStateChanged.RemoveListener(OnPepperNPCStateChanged);
            }
            
            if (pepperController.onNPCTaskChanged != null)
            {
                pepperController.onNPCTaskChanged.RemoveListener(OnPepperNPCTaskChanged);
            }
        }
    }
    
    // Event handlers for Pepper
    private void OnPepperStateChanged(PepperController.PepperState newState)
    {
        Debug.Log($"<color={GetPepperStateColor(newState)}>[Pepper State] State changed to: <b>{newState}</b></color>");
        UpdateDisplay();
    }
    
    private void OnPepperActionPerformed(PepperController.AgentAction action)
    {
        Debug.Log($"[Pepper Action] Performing action: <b>{action}</b>");
        UpdateDisplay();
    }
    
    private void OnPepperNPCStateChanged(NPCController.NPCState npcState)
    {
        Debug.Log($"[Pepper Monitoring] NPC state observed: <b>{npcState}</b>");
        UpdateDisplay();
    }
    
    private void OnPepperNPCTaskChanged(NPCTask npcTask)
    {
        if (npcTask != null)
        {
            Debug.Log($"[Pepper Monitoring] NPC task observed: <b>{npcTask.taskName}</b>");
        }
        UpdateDisplay();
    }
    
    // Event handlers for NPC (kept from original)
    private void OnNPCStateChanged(NPCController.NPCState newState)
    {
        Debug.Log($"<color={GetNPCStateColor(newState)}>[NPC State] State changed to: <b>{newState}</b></color>");
        UpdateDisplay();
    }
    
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
                
            string color = GetNPCStateColor(npcController.CurrentState);
            npcLogText.text = $"<color={color}>NPC State: {stateText}</color>\n" +
                            $"Task: {taskText}";
        }
        
        if (pepperController != null && pepperLogText != null)
        {
            string stateText = pepperController.CurrentState.ToString();
            string stateColor = GetPepperStateColor(pepperController.CurrentState);
            string description = pepperController.GetCurrentStateDescription();
            string actionDescription = pepperController.GetCurrentActionDescription();
            
            pepperLogText.text = $"<color={stateColor}>Pepper State: {stateText}</color>\n" +
                               $"{description}\n" +
                               $"Action: {actionDescription}\n";
        }
    }
    
    private string GetNPCStateColor(NPCController.NPCState state)
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
            case PepperController.PepperState.MonitoringNPC:
                return "green";
            default:
                return "white";
        }
    }
}