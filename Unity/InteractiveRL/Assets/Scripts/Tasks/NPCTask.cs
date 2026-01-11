using UnityEngine;

namespace Tasks
{
    [CreateAssetMenu(fileName = "NPCTask", menuName = "Scriptable Objects/NPCTask")]
    public class NPCTask : ScriptableObject
    {
        [Header("Task Information")]
        public string taskName;
        [Header("Target Information")]
        public string targetObjectName; 
        public float acceptanceRadius = 1f;
    
        [Header("NPC Behavior")]
        public string animationName;
        public float taskDuration = 3f;
    }
}
