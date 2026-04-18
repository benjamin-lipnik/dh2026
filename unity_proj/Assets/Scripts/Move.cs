using UnityEngine;
using System;

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
	public Transform body_low_origin;
	public GameObject fireball;

	float nextPunchTime = 0f;
	float punchCooldown = 0.4f; // seconds between punches (adjust)

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
            (Input.GetKey(KeyCode.E) ? 1f : 0f) -
            (Input.GetKey(KeyCode.Q) ? 1f : 0f);

        // -------- EXTERNAL INPUT (IC) --------
		float boost = (Input.GetKey(KeyCode.LeftShift)?1.0f:0.0f);
        if (ic?.msg != null)
        {
			boost += ((ic.msg.boost_forward)?1.0f:0.0f);
            forward += ic.msg.move_z;
            right   += ic.msg.move_x;
            up      += ic.msg.move_y;
            yaw     += ic.msg.turn;

			// "uppercut"
			// "direct"
			// "hook"

			bool wantsPunch =
			    Input.GetKey(KeyCode.Space) ||
			    ic.msg.punch_left == "uppercut" ||
			    ic.msg.punch_left == "hook" ||
			    ic.msg.punch_left == "direct" ||
			    ic.msg.punch_right == "uppercut" ||
			    ic.msg.punch_right == "hook" ||
			    ic.msg.punch_right == "direct";

			if (wantsPunch && Time.time >= nextPunchTime)
			{
			    nextPunchTime = Time.time + punchCooldown;

			    ic.msg.punch_right = "null";
			    ic.msg.punch_left = "null";

			    myAnimator.SetTrigger("Punch");

			    Vector3 fb_pos = transform.position + transform.forward * 2.0f + new Vector3(0, 1, 0);
			    GameObject fb = GameObject.Instantiate(fireball, fb_pos, Quaternion.identity);

			    Rigidbody r = fb.GetComponent<Rigidbody>();
			    r.velocity = rb.velocity + transform.forward * 20;

			    GameObject.Destroy(fb, 5);
			}

        }
		if(Math.Abs(forward) > 0.3f) {
            myAnimator.SetBool("isFlying", true);
		}else {
            myAnimator.SetBool("isFlying", false);
		}

		forward *= (1.0f + boost);
		right   *= (1.0f + boost);
		up      *= (1.0f + boost);
		// yaw     *= (1.0f + boost);

        // -------- MOVEMENT --------
		Vector3 move_xy =
			(transform.forward * forward) +
       		(transform.right * right);

        Vector3 velocity =
            move_xy* fwdSpeed +
            (transform.up * up) * upDownSpeed;

        rb.velocity = velocity;
		body_low_origin.localRotation = Quaternion.Euler(0,-right*10, 0);

        // -------- ROTATION --------
        Vector3 torque = new Vector3(
            0f,
            yaw * yawSpeed,
            0f
        );

        rb.angularVelocity = torque * Time.fixedDeltaTime;
    }
}
