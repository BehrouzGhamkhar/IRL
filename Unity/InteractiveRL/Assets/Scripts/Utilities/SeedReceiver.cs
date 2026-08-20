using Managers;
using Unity.MLAgents;
using UnityEngine;

namespace Utilities
{
    public class SeedReceiver : MonoBehaviour
    {
        [SerializeField] private CommunicationManager communicationManager;
    
        void Awake()
        {
            // EnvironmentParameters returns float, so we need to cast to int
            float seedFloat = Academy.Instance.EnvironmentParameters.GetWithDefault("npc_seed", 42f);
            int seed = Mathf.FloorToInt(seedFloat);
            if (communicationManager == null)
            {
                communicationManager = GetComponent<CommunicationManager>();
                Debug.Log($"CommunicationManager: {communicationManager}");
            }
        
            if (communicationManager != null)
            {
                communicationManager.SetMasterSeed(seed);
                Debug.Log($"[SeedReceiver] Received seed from Python: {seed}");
            }
        }
    }
}