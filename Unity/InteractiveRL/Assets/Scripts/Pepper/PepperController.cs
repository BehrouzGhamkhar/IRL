using UnityEngine;
using System.Collections;
using System;
using DG.Tweening;

public class PepperController : MonoBehaviour
{
    [SerializeField] private Animator robotAnimator;
    [SerializeField] private Transform headBone;
    [SerializeField] private float headRotationSpeed = 5f;
    [SerializeField] private float rotationSpeed = 2f;
    [SerializeField] private float lookAtDuration = 3f;
    

    private Transform currentLookTarget;
    private float lookEndTime;
    private bool isLooking;

    public enum AgentAction
    {
        DoNothing = 0,
        Wait = 1,
        Look = 2,
        Wave = 3,
        HandShake = 6
    };

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

    void Update()
    {
        // Handle continuous look behavior
        if (isLooking && Time.time < lookEndTime && currentLookTarget != null)
        {
            Vector3 lookDirection = currentLookTarget.position - headBone.position;
            Quaternion targetRotation = Quaternion.LookRotation(lookDirection);
            headBone.rotation = Quaternion.Slerp(
                headBone.rotation, 
                targetRotation, 
                Time.deltaTime * headRotationSpeed
            );
        }
        else if (isLooking && Time.time >= lookEndTime)
        {
            isLooking = false;
        }

        HandleKeyboardInput();
    }

    private void ExecuteAction(AgentAction rAction)
    {
        switch (rAction)
        {
            case AgentAction.Wait:
                ActionWait();
                break;
                
            case AgentAction.Look:
                ActionLook();
                break;
                
            case AgentAction.Wave:
                ActionLook(); // Look first
                ActionWave();
                break;
                
            case AgentAction.HandShake:
                float tryHandShakeTime = 2.0f;
                ActionLook(); // Look first
                StartCoroutine(ActionHandshake(tryHandShakeTime));
                break;
                
            case AgentAction.DoNothing:
                // Intentionally blank
                break;
                
            default:
                Debug.LogWarning($"Unhandled action: {rAction}");
                break;
        }
    }

    #region Action Implementations
    
    private void ActionWait()
    {
        robotAnimator.SetTrigger("Idle");
    }

    private void ActionLook()
    {
        var closestPerson = FindNearestPerson();
        currentLookTarget = closestPerson.transform.Find("HeadPosition");

        if (currentLookTarget != null)
        {
            isLooking = true;
            lookEndTime = Time.time + lookAtDuration;
        }
    }

    private void ActionWave()
    {
        robotAnimator.SetTrigger("Wave");
    }

    IEnumerator ActionHandshake(float delayTime)
    {
        robotAnimator.SetTrigger("TryHandshake");
        var closestPerson = FindNearestPerson();
        yield return new WaitForSeconds(delayTime);
        if (closestPerson != null)
        {
            Vector3 targetPosition = closestPerson.position;
            if (Vector3.Distance(transform.position, targetPosition) < 2.0f)
            {
                robotAnimator.SetTrigger("Handshake");
            }
            else
            {
                Debug.LogWarning("Too far to handshake.");
                robotAnimator.SetTrigger("Idle");
            }
        }
        else
        {
            Debug.LogWarning("No person found to handshake with.");
            robotAnimator.SetTrigger("Idle");
        }
    }
    
    #endregion

    private Transform FindNearestPerson()
    {
        GameObject[] people = GameObject.FindGameObjectsWithTag("Person");
        float closestDistance = float.MaxValue;
        Transform closestPerson = null;

        if (people.Length == 0)
        {
            Debug.LogWarning("No person found to look at.");
            return closestPerson;
        }

        foreach (var person in people)
        {
            float distance = Vector3.Distance(transform.position, person.transform.position);
            if (distance < closestDistance)
            {
                closestDistance = distance;
                closestPerson = person.transform;
            }
        }
        return closestPerson;
    }
    private void HandleKeyboardInput()
    {
        if (Input.GetKeyDown(KeyCode.I) || Input.GetKeyDown(KeyCode.Alpha0))
        {
            ExecuteAction(AgentAction.Wait);
        }
        else if (Input.GetKeyDown(KeyCode.W) || Input.GetKeyDown(KeyCode.Alpha1))
        {
            ExecuteAction(AgentAction.DoNothing);
        }
        else if (Input.GetKeyDown(KeyCode.H) || Input.GetKeyDown(KeyCode.Alpha2))
        {
            ExecuteAction(AgentAction.Look);
        }
        else if (Input.GetKeyDown(KeyCode.L) || Input.GetKeyDown(KeyCode.Alpha3))
        {
            ExecuteAction(AgentAction.Wave);
        }
        else if (Input.GetKeyDown(KeyCode.R) || Input.GetKeyDown(KeyCode.Alpha4))
        {
            ExecuteAction(AgentAction.HandShake);
        }
    }
}