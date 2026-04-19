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
	public bool keyboardInput = false;

    private Rigidbody rb;
    public Animator myAnimator;
	public Transform body_low_origin;
	public GameObject fireball;
	public Transform opponent;
	public Transform origin;
	public Collider myCollider;

	float nextPunchTime = 0f;
	float punchCooldown = 0.4f; // seconds between punches (adjust)

	[Header("Aim Assist")]
	public float aimAssistDrop = 3f;
	public float aimAssistStrength = 0.001f;   // how fast it rotates toward enemy
	public float aimAssistBaseYaw = 0.001f;   // how fast it rotates toward enemy
	public float aimAssistBaseUp  = 0.01f;    // how fast it adjusts vertical movement0
	public float coastStrength = 0.5f; // tweak this

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        rb.useGravity = false;
    }

	float sign(float x) {
		return (x < 0)?-1:1;
	}
	float clamp(float x, float min, float max) {
		if(x<min)
			return min;
		if(x>max)
			return max;
		return x;
	}

    void FixedUpdate()
    {
		float en = ((keyboardInput)?1.0f:0.0f);

        // -------- INPUT --------
        float forward = Input.GetAxis("Vertical") * en;
        float yaw = Input.GetAxis("Horizontal") * en;

        float up =
            ((Input.GetKey(KeyCode.K) ? 1f : 0f) -
            (Input.GetKey(KeyCode.J) ? 1f : 0f)) * en;

        float right =
            ((Input.GetKey(KeyCode.E) ? 1f : 0f) -
            (Input.GetKey(KeyCode.Q) ? 1f : 0f)) * en;

        // -------- EXTERNAL INPUT (IC) --------
		float boost = (Input.GetKey(KeyCode.LeftShift)?1.0f:0.0f) * en;
        if (ic?.msg != null)
        {
			boost    = ((ic.msg.boost_forward)?1.0f:0.0f);
            forward += ic.msg.move_z;
            right   += ic.msg.move_x;
            up      += ic.msg.move_y;
            yaw     += ic.msg.turn;

			// "uppercut"
			// "direct"
			// "hook"

			bool wantsPunch =
			    (Input.GetKey(KeyCode.Space) && keyboardInput) ||
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

			    Vector3 fb_pos = transform.position + transform.forward + new Vector3(0, 1.0f, 0);
			    GameObject fb = GameObject.Instantiate(fireball, fb_pos, Quaternion.identity);
				Fireball fbs = fb.GetComponent<Fireball>();
				fbs.skip_collisions = myCollider;

			    Rigidbody r = fb.GetComponent<Rigidbody>();
			    r.velocity = rb.velocity + transform.forward * 10;
			}

        }
		if(Math.Abs(forward) > 0.3f) {
            myAnimator.SetBool("isFlying", true);
		}else {
            myAnimator.SetBool("isFlying", false);
		}

		// forward *= (1.0f + boost);
		// right   *= (1.0f + boost);
		// up      *= (1.0f + boost);
		// yaw     *= (1.0f + boost);

		if(opponent != null)
		{
		    Vector3 toOpponent = opponent.position - transform.position;
		    Vector3 dir = toOpponent.normalized;
		    Vector3 fforward = transform.forward;

			// float strength = aimAssistStrength / Mathf.Pow(toOpponent.magnitude, aimAssistDrop);
			float strength = aimAssistStrength / Mathf.Pow(toOpponent.magnitude, aimAssistDrop);

		    // signed angle around Y axis
		    float angle = Vector3.SignedAngle(fforward, dir, Vector3.up);
			if(Math.Abs(angle) < 10f) {
				angle = 0;
			}
		    yaw += angle * 0.001f * aimAssistBaseYaw * strength;

		    float verticalDiff = toOpponent.y;
		    up += verticalDiff * aimAssistBaseUp * strength;
		}

		if(origin != null)
		{
		    Vector3 toOrigin = origin.position - transform.position;

		    // Normalize so it only gives direction
		    Vector3 dir = toOrigin.normalized;

		    // -------- CONVERT TO LOCAL AXES --------
		    float forwardAssist = Vector3.Dot(dir, transform.forward);
		    float rightAssist   = Vector3.Dot(dir, transform.right);
		    float upAssist      = Vector3.Dot(dir, transform.up);

		    // -------- APPLY SOFT COASTING --------
		    forward += forwardAssist * coastStrength;
		    right   += rightAssist   * coastStrength;
		    up      += upAssist      * coastStrength;
		}

		right = clamp(right, -1f, 1f);
		up = clamp(up, -1f, 1f);
		yaw = clamp(yaw, -1f, 1f);

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
