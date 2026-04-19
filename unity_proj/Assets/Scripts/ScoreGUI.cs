using UnityEngine;

public class ScoreGUI : MonoBehaviour
{
	public int[] scores = {0,0};

    public GUIStyle style;

	void Awake()
	{
	    style = new GUIStyle();
	    style.fontSize = 20;
	    style.alignment = TextAnchor.MiddleCenter;
	    style.normal.textColor = Color.white;
	}

    void OnGUI()
    {
        float screenW = Screen.width;
        float screenH = Screen.height;

        float y = screenH * 0.1f;
        float boxW = 200f;
        float boxH = 50f;

        // -------- LEFT SCORE (1st quarter) --------
        Rect leftRect = new Rect(
            (screenW * 0.25f) - (boxW * 0.5f),
            y,
            boxW,
            boxH
        );

        GUI.Label(leftRect, scores[0].ToString(), style);

        // -------- RIGHT SCORE (3rd quarter) --------
        Rect rightRect = new Rect(
            (screenW * 0.75f) - (boxW * 0.5f),
            y,
            boxW,
            boxH
        );

        GUI.Label(rightRect, scores[1].ToString(), style);
    }
}
