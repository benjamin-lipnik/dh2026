using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class Move : MonoBehaviour
{
    [Header("Movement")]
    public float fwdSpeed = 15f;
	public float upDownSpeed = 10f;

    [Header("Rotation")]
    public float yawSpeed = 120f;

    public InputController ic;

    private Rigidbody rb;
    public Animator myAnimator;

	public enum CharacterState
	{
	    Idle = 0,
	    Flying
	}
	public CharacterState cs = CharacterState.Idle;

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
		float boost = (Input.GetKey(KeyCode.LeftShift)?1.0f:0.0f);
        if (ic?.msg != null)
        {
			boost += ((ic.msg.boost_forward)?1.0f:0.0f);
            forward += ic.msg.move_z;
            right   += ic.msg.move_x;
            up      += ic.msg.move_y;
            yaw     += ic.msg.turn;
        }
		if(forward != 0f) {
			cs = CharacterState.Flying;
            myAnimator.SetBool("isFlying", true);
		}else {
			cs = CharacterState.Idle;
            myAnimator.SetBool("isFlying", false);
		}

		forward *= (1.0f + boost);
		right   *= (1.0f + boost);
		up      *= (1.0f + boost);
		// yaw     *= (1.0f + boost);

        // -------- MOVEMENT --------
        Vector3 velocity =
            (transform.forward * forward) * fwdSpeed +
            (transform.right * right) * fwdSpeed +
            (transform.up * up) * upDownSpeed;

        rb.velocity = velocity;

        // -------- ROTATION --------
        Vector3 torque = new Vector3(
            0f,
            yaw * yawSpeed,
            0f
        );

        rb.angularVelocity = torque * Time.fixedDeltaTime;
    }
}
