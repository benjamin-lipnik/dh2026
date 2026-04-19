using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DoHit : MonoBehaviour
{
	public Collider opponent_collider;
	public bool opponent_in_range;

	void OnTriggerEnter(Collider collider) {
		if(collider == opponent_collider) {
			opponent_in_range = true;
		}
	}
	void OnTriggerExit(Collider collider) {
		if(collider == opponent_collider) {
			opponent_in_range = false;
		}
	}
}
