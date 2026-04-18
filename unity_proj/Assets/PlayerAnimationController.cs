using UnityEngine;

public class PlayerAnimationController : MonoBehaviour
{
    // This creates a reference to your Animator "brain"
    private Animator myAnimator;

    void Start()
    {
        // When the game starts, this finds the Animator on your character
        myAnimator = GetComponent<Animator>();
    }

    void Update()
    {
        // 1. FLYING (Hold Space to fly, release to stop)
        
        // If the spacebar is pushed DOWN this frame, turn flying ON
        if (Input.GetKeyDown(KeyCode.Space))
        {
            myAnimator.SetBool("isFlying", true); 
        }
        // If the spacebar is let GO this frame, turn flying OFF
        else if (Input.GetKeyUp(KeyCode.Space))
        {
            myAnimator.SetBool("isFlying", false); 
        }

        // 2. PUNCHING (Press X to punch)
        if (Input.GetKeyDown(KeyCode.X))
        {
            myAnimator.SetTrigger("Punch");
        }

        // 3. BOXING (Press Y to box)
        if (Input.GetKeyDown(KeyCode.Y))
        {
            myAnimator.SetTrigger("Box");
        }
    }
}