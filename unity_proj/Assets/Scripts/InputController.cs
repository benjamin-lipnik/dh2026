using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

[Serializable]
public class InputMessage
{
    public double t;

    public float move_x;
    public float move_y;
    public float move_z;

    public float turn;

    public bool boost_forward;
    public bool boost_backward;
    public bool boost_armed;
    public bool boost_needs_guard;

    // nullable fields must be wrapped or made string/object
    public string punch_left;
    public string punch_right;
}

public class InputController : MonoBehaviour
{
    private UdpClient udpClient;
    private Thread listenThread;
    private bool running = false;

    public int port = 55555;
	public InputMessage msg = null;

    void Start()
    {
        udpClient = new UdpClient(port);
        running = true;

        listenThread = new Thread(Listen);
        listenThread.IsBackground = true;
        listenThread.Start();
        Debug.Log("UDP Listener started on port " + port);
    }

    private void Listen()
    {
        IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, port);

        while (running)
        {
            try
            {
                byte[] data = udpClient.Receive(ref remoteEndPoint);
                string message = Encoding.UTF8.GetString(data);

                // Unity API must run on main thread, so we queue it
                UnityMainThreadDispatcher.Enqueue(() =>
                {
                	msg = JsonUtility.FromJson<InputMessage>(message);
					// Debug.Log(message);
                });
            }
            catch
            {

                // Debug.LogError("UDP error: " + e.Message);
            }
        }
    }

    void OnApplicationQuit()
    {
        running = false;

        udpClient?.Close();

        if (listenThread != null && listenThread.IsAlive)
            listenThread.Abort();
    }
}
