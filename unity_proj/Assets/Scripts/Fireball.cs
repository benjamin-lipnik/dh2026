using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Fireball : MonoBehaviour
{
	public Collider skip_collisions;

    // Start is called before the first frame update
    void Start()
    {
		GameObject.Destroy(this.gameObject, 5);
    }

    // Update is called once per frame
    void Update()
    {

    }

	void OnCollisionEnter(Collision collision)
	{
		if(skip_collisions != null) {
			if(collision.collider != skip_collisions) {
				// Debug.Log(collision.collider.gameObject.name);
				Destroy(this.gameObject);
			}
		}
		else {
			Destroy(this.gameObject);
		}
	}
}
