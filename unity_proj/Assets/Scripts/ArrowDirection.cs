using UnityEngine;

public class ArrowDirection : MonoBehaviour
{
    [Header("References")]
    public Transform from;
    public Transform to;

    void Update()
    {
        if (from == null || to == null)
            return;

        Vector3 dir = to.position - from.position;

        if (dir.sqrMagnitude < 0.0001f)
            return;

        transform.rotation = Quaternion.LookRotation(dir.normalized, Vector3.up);
    }
}
