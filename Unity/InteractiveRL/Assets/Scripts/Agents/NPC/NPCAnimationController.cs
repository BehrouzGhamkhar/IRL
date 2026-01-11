using System.Collections;
using UnityEngine;
using DG.Tweening;

namespace Agents.NPC
{
    public class NPCAnimationController : MonoBehaviour
    {
        [SerializeField] private Animator npcAnimator;
        
        private const string IS_WALKING_BOOL = "IsWalking";
        private const string IS_IDLE_BOOL = "IsIdle";
    
        void Start()
        {
            if (npcAnimator == null)
            {
                npcAnimator = GetComponent<Animator>();
                if (npcAnimator == null)
                {
                    Debug.LogError("NPC Animator not found!");
                }
            }
        }
        
        public void PlayWalk()
        {
            if (npcAnimator != null)
            {
                ResetAllBools();
                npcAnimator.SetBool(IS_WALKING_BOOL, true);
            }
        }
    
        public void PlayIdle()
        {
            if (npcAnimator != null)
            {
                ResetAllBools();
                npcAnimator.SetBool(IS_IDLE_BOOL, true);
            }
        }
    
        public void PlayTaskAnimation(string animationName)
        {
            if (npcAnimator != null)
            {
                Debug.Log($"Playing animation clip: {animationName}");
                npcAnimator.Play(animationName);
            }
        }
        
        public IEnumerator RotateToTargetCoroutine(Transform target, float duration = 0.5f)
        {
            if (target != null)
            {
                transform.DORotate(target.eulerAngles, duration, RotateMode.Fast);
                yield return new WaitForSeconds(duration);
            }
        }
        
        public bool IsAnimationPlaying(string animationName)
        {
            if (npcAnimator == null) return false;
        
            AnimatorStateInfo stateInfo = npcAnimator.GetCurrentAnimatorStateInfo(0);
            return stateInfo.IsName(animationName);
        }
        
        private void ResetAllBools()
        {
            if (npcAnimator != null)
            {
                npcAnimator.SetBool(IS_WALKING_BOOL, false);
                npcAnimator.SetBool(IS_IDLE_BOOL, false);
            }
        }
        
        public void SetAnimationSpeed(float speed)
        {
            if (npcAnimator != null)
            {
                npcAnimator.speed = speed;
            }
        }
        
        public void ResetAnimationSpeed()
        {
            SetAnimationSpeed(1.0f);
        }
    }
}