using UnityEngine.Events;

namespace Managers.Reward
{
    // Simple interface for all reward providers
    public interface IRewardProvider
    {
        bool IsEnabled { get; set; }
        UnityEvent<float> OnReward { get; }
        void Reset();
    }
}