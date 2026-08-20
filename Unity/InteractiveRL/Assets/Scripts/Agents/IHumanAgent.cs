using Tasks;
using UnityEngine;
using UnityEngine.Events;
using static Agents.NPC.NPCController;

namespace Agents
{
    /// <summary>
    /// Shared contract for anything that plays the "human" role in a training episode —
    /// either the AI-driven NPCController or a real person using VRPersonController.
    /// CommunicationManager talks to this interface exclusively, so it never needs to
    /// know which concrete type is in the scene.
    /// </summary>
    public interface IHumanAgent
    {
        // Current state 
        NPCState    CurrentState { get; }
        NPCTask     CurrentTask  { get; }
        Transform   Transform    { get; }

        // Events 
        UnityEvent<NPCState> OnStateChanged { get; }
        UnityEvent<NPCTask>  OnTaskChanged  { get; }

        // Episode control 
        void ForceStartTask();
        void ClearCurrentTask();
        void StopMovement();
    }
}