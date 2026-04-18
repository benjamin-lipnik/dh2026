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

	float sign(float x) {
		return (x < 0)?-1:1;
	}

	float apply_deadzone(float v) {
	    return (Math.Abs(v) < 0.25) ? 0.0f : v*v*sign(v);
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
                	InputMessage new_msg = JsonUtility.FromJson<InputMessage>(message);

					msg.t = new_msg.t;

					msg.move_x = apply_deadzone(new_msg.move_x);
					msg.move_y = apply_deadzone(new_msg.move_y);
					msg.move_z = apply_deadzone(new_msg.move_z);
					msg.turn = apply_deadzone(new_msg.turn);

					msg.boost_forward = new_msg.boost_forward;
					msg.boost_backward = new_msg.boost_backward;
					msg.boost_armed = new_msg.boost_armed;
					msg.boost_needs_guard = new_msg.boost_needs_guard;

					if (new_msg.punch_left != "null")
					{
					    msg.punch_left = new_msg.punch_left;
					}
					if (new_msg.punch_right != "null")
					{
					    msg.punch_right = new_msg.punch_right;
					}
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
