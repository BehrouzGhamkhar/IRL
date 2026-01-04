using System.Collections;
using UnityEngine;

namespace Pepper
{
    public class PepperAnimationController : MonoBehaviour
    {
        [SerializeField] private Animator robotAnimator;
    
        // Animation parameter names (should match your Animator parameters)
        private const string IDLE_TRIGGER = "Idle";
        private const string WAVE_TRIGGER = "Wave";
        private const string TRY_HANDSHAKE_TRIGGER = "TryHandshake";
        private const string HANDSHAKE_TRIGGER = "Handshake";
    
        void Start()
        {
            if (robotAnimator == null)
            {
                robotAnimator = GetComponent<Animator>();
                if (robotAnimator == null)
                {
                    Debug.LogError("Robot Animator not found!");
                }
            }
        }
    
        // Public methods to trigger animations
        public void PlayIdle()
        {
            if (robotAnimator != null)
            {
                robotAnimator.SetTrigger(IDLE_TRIGGER);
            }
        }
    
        public void PlayWave()
        {
            if (robotAnimator != null)
            {
                robotAnimator.SetTrigger(WAVE_TRIGGER);
            }
        }
    
        public void PlayHandshake()
        {
            if (robotAnimator != null)
            {
                robotAnimator.SetTrigger(HANDSHAKE_TRIGGER);
            }
        }
    
        public void PlayTryHandshake()
        {
            if (robotAnimator != null)
            {
                robotAnimator.SetTrigger(TRY_HANDSHAKE_TRIGGER);
            }
        }
    
        // Coroutine for handshake with person proximity check
        public IEnumerator PlayHandshakeWithProximityCheck(float delayTime, Transform personTransform = null)
        {
            PlayTryHandshake();
        
            yield return new WaitForSeconds(delayTime);
        
            if (personTransform != null)
            {
                if (Vector3.Distance(transform.position, personTransform.position) < 2.0f)
                {
                    PlayHandshake();
                }
                else
                {
                    Debug.LogWarning("Too far to handshake.");
                    PlayIdle();
                }
            }
            else
            {
                Debug.LogWarning("No person found to handshake with.");
                PlayIdle();
            }
        }
    
        // Method to check if animation is playing
        public bool IsAnimationPlaying(string animationName)
        {
            if (robotAnimator == null) return false;
        
            AnimatorStateInfo stateInfo = robotAnimator.GetCurrentAnimatorStateInfo(0);
            return stateInfo.IsName(animationName);
        }
    
        // Reset all triggers to prevent conflicts
        public void ResetAllTriggers()
        {
            if (robotAnimator != null)
            {
                robotAnimator.ResetTrigger(IDLE_TRIGGER);
                robotAnimator.ResetTrigger(WAVE_TRIGGER);
                robotAnimator.ResetTrigger(HANDSHAKE_TRIGGER);
                robotAnimator.ResetTrigger(TRY_HANDSHAKE_TRIGGER);
            }
        }
    
        // Set animation speed
        public void SetAnimationSpeed(float speed)
        {
            if (robotAnimator != null)
            {
                robotAnimator.speed = speed;
            }
        }
    
        // Reset animation speed to normal
        public void ResetAnimationSpeed()
        {
            SetAnimationSpeed(1.0f);
        }
    }
}

