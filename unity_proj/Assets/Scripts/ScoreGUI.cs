using UnityEngine;

public class ScoreGUI : MonoBehaviour
{
	public int[] scores = {0,0};

    public GUIStyle style;

	void Awake()
	{
	    style = new GUIStyle();
	    style.fontSize = 20;
	    // style.alignment = TextAnchor.MiddleCenter;
	    style.normal.textColor = Color.white;
	}

    void OnGUI()
	{
	    float screenW = Screen.width;
	    float screenH = Screen.height;

	    float y = 30f;
	    float boxW = 250f;
	    float boxH = 50f;

	    // -------- LEFT SCORE (left-aligned on left side) --------
	    Rect leftRect = new Rect(
	        10f,   // small padding from left edge
	        y,
	        boxW,
	        boxH
	    );

	    GUI.Label(leftRect, "Score: " + scores[0].ToString(), style);

	    // -------- RIGHT SCORE (left-aligned on right half) --------
	    Rect rightRect = new Rect(
	        (screenW * 0.5f) + 10f,  // start of right half + padding
	        y,
	        boxW,
	        boxH
	    );

	    GUI.Label(rightRect, "Score: " + scores[1].ToString(), style);
	}
}
