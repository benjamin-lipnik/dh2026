using UnityEngine;

public class CubeSpawner : MonoBehaviour
{
    public GameObject cubePrefab;   // assign in inspector
    public int count = 100;
    public float radius = 5f;

    void Start()
    {
        SpawnCubes();
    }

    void SpawnCubes()
    {
        for (int i = 0; i < count; i++)
        {
            // random position around this object
            Vector3 randomOffset = new Vector3(
                Random.Range(-radius, radius),
                Random.Range(0f, radius),   // optional: more spread upward
                Random.Range(-radius, radius)
            );

            Vector3 spawnPos = transform.position + randomOffset;

            // random rotation
            Quaternion spawnRot = Random.rotation;

            Instantiate(cubePrefab, spawnPos, spawnRot);
        }
    }
}
