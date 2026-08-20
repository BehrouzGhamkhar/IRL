using UnityEngine;

namespace Tasks
{
    [CreateAssetMenu(fileName = "NPCTask", menuName = "Scriptable Objects/NPCTask")]
    public class NPCTask : ScriptableObject
    {
        [Header("Task Information")]
        public int id;
        public string taskName;

        [Header("Target Information")]
        [Tooltip("True = random NavMesh point near robot each episode (breaks distance memorisation).\n" +
                 "False = fixed target object position with small noise.")]
        public bool randomPosition = false;
        public string targetObjectName;
        public float acceptanceRadius = 1f;

        [Tooltip("Only used when randomPosition = false. Small XZ offset added to the fixed target.")]
        public float positionNoise = 0.15f;

        [Tooltip("Only used when randomPosition = true.")]
        public float minRange = 1f;
        public float maxRange = 5f;

        [Header("Observation Timing")]
        [Tooltip("Seconds to wait into the animation before collecting observations.")]
        public float observationDelay = 1f;
        [Tooltip("Random ± variation on the delay each episode so agent doesn't overfit to one moment.")]
        public float observationDelayNoise = 0.2f;

        [Header("NPC Behavior")]
        public string animationName;
        public float taskDuration = 3f;
    }
}