using UnityEngine;
using UnityEngine.SceneManagement;

public class MainMenuManager : MonoBehaviour
{
    [Header("UI Panels")]
    public GameObject tutorialWindow;

    [Header("Tutorial Slides")]
    public GameObject[] slides; 
    public GameObject nextButton; // We need to tell the script where this is
    public GameObject backButton; // We need to tell the script where this is

    private int currentSlideIndex = 0;

    void Start()
    {
        tutorialWindow.SetActive(false);
    }

    public void StartGame()
    {
        SceneManager.LoadScene("MainScene"); 
    }

    public void OpenTutorial()
    {
        tutorialWindow.SetActive(true);
        currentSlideIndex = 0; // Always start on Slide 1
        UpdateSlideVisibility();
    }

    public void CloseTutorial()
    {
        tutorialWindow.SetActive(false);
    }

    public void NextSlide()
    {
        if (currentSlideIndex < slides.Length - 1)
        {
            currentSlideIndex++; 
            UpdateSlideVisibility();
        }
    }

    // NEW: Function to go backward
    public void PreviousSlide()
    {
        if (currentSlideIndex > 0)
        {
            currentSlideIndex--; 
            UpdateSlideVisibility();
        }
    }

    private void UpdateSlideVisibility()
    {
        // 1. Turn the right slide on, and turn the others off
        for (int i = 0; i < slides.Length; i++)
        {
            slides[i].SetActive(i == currentSlideIndex);
        }

        // 2. Hide "Back" on the first slide (Index 0). Show it everywhere else.
        backButton.SetActive(currentSlideIndex > 0);

        // 3. Hide "Next" on the last slide. Show it everywhere else.
        nextButton.SetActive(currentSlideIndex < slides.Length - 1);
    }
}