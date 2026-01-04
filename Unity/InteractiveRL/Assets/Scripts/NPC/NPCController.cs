using UnityEngine;
using UnityEngine.AI;

public class NPCController: MonoBehaviour
{
    public float moveRadius = 5f;
    public float newPointDelay = 3f;
    
    private NavMeshAgent agent;
    private Vector3 centerPoint;
    private float timer;
    
    void Start()
    {
        agent = GetComponent<NavMeshAgent>();
        centerPoint = transform.position;
        GetNewDestination();
    }
    
    void Update()
    {
        timer += Time.deltaTime;
        
        // Get new destination every few seconds
        if (timer >= newPointDelay)
        {
            GetNewDestination();
            timer = 0;
        }
    }
    
    void GetNewDestination()
    {
        Vector3 randomPoint = centerPoint + Random.insideUnitSphere * moveRadius;
        randomPoint.y = centerPoint.y; // Keep same height
        
        NavMeshHit hit;
        if (NavMesh.SamplePosition(randomPoint, out hit, moveRadius, NavMesh.AllAreas))
        {
            agent.SetDestination(hit.position);
        }
    }
}