using System;
using System.Collections.Concurrent;
using UnityEngine;

public class UnityMainThreadDispatcher : MonoBehaviour
{
    private static readonly ConcurrentQueue<Action> actions = new ConcurrentQueue<Action>();

    public static void Enqueue(Action action)
    {
        actions.Enqueue(action);
    }

    void Update()
    {
        while (actions.TryDequeue(out var action))
        {
            action?.Invoke();
        }
    }
}
