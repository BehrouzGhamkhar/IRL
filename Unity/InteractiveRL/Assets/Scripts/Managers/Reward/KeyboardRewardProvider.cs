using UnityEngine;
using UnityEngine.Events;
using Utilities;

namespace Managers.Reward
{
    // Experiment 2: Human gives rewards using keyboard arrows

    public class KeyboardRewardProvider : MonoBehaviour, IRewardProvider
    {
        [Header("Keys")]
        public KeyCode goodKey = KeyCode.UpArrow;
        public KeyCode badKey = KeyCode.DownArrow;

        [Header("Reward Values")]
        public float goodReward = 1f;
        public float badReward = -1f;

        [Header("Cooldown")]
        public float cooldownSeconds = 0.5f;

        public bool IsEnabled { get; set; } = true;
        public UnityEvent<float> OnReward { get; } = new UnityEvent<float>();

        private float lastRewardTime = -999f;

        private void Update()
        {
            if (!IsEnabled) return;
            if (Time.time - lastRewardTime < cooldownSeconds) return;

            if (Input.GetKeyDown(goodKey))
            {
                lastRewardTime = Time.time;
                FeedbackLogger.Add("Keyboard", goodReward);
                OnReward?.Invoke(goodReward);
                Debug.Log($"[Keyboard] +{goodReward}");
            }
            else if (Input.GetKeyDown(badKey))
            {
                lastRewardTime = Time.time;
                FeedbackLogger.Add("Keyboard", badReward);
                OnReward?.Invoke(badReward);
                Debug.Log($"[Keyboard] {badReward}");
            }
        }

        public void Reset()
        {
            lastRewardTime = -999f;
        }
    }
}