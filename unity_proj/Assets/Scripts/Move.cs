using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class Move : MonoBehaviour
{
    [Header("Movement")]
    public float fwdSpeed = 15f;

    [Header("Rotation")]
    public float yawSpeed = 120f;

    public InputController ic;

    private Rigidbody rb;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        rb.useGravity = false;
    }

    void FixedUpdate()
    {
        // -------- INPUT --------
        float forward = Input.GetAxis("Vertical");
        float yaw = Input.GetAxis("Horizontal");

        float up =
            (Input.GetKey(KeyCode.K) ? 1f : 0f) -
            (Input.GetKey(KeyCode.J) ? 1f : 0f);

        float right =
            (Input.GetKey(KeyCode.Q) ? 1f : 0f) -
            (Input.GetKey(KeyCode.E) ? 1f : 0f);

        // -------- EXTERNAL INPUT (IC) --------
        if (ic?.msg != null)
        {
            forward += ic.msg.move_z * (1.0f + ((ic.msg.boost_forward)?1.0f:0.0f));
            right   += ic.msg.move_x * (1.0f + ((ic.msg.boost_forward)?1.0f:0.0f));
            up      += ic.msg.move_y * (1.0f + ((ic.msg.boost_forward)?1.0f:0.0f));
            yaw     += ic.msg.turn;
        }

        // -------- MOVEMENT --------
        Vector3 velocity =
            (transform.forward * forward) +
            (transform.right * right) +
            (transform.up * up);

        rb.velocity = velocity * fwdSpeed;

        // -------- ROTATION --------
        Vector3 torque = new Vector3(
            0f,
            yaw * yawSpeed,
            0f
        );

        rb.angularVelocity = torque * Time.fixedDeltaTime;
    }
}
