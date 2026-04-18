using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class UDPJump : MonoBehaviour
{
    private UdpClient udpClient;
    private Thread listenThread;
    private bool running = false;

    public int port = 55555;
	public Rigidbody rb;

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

                if (message.Contains("jump"))
                {
                    // Unity API must run on main thread, so we queue it
                    UnityMainThreadDispatcher.Enqueue(() =>
                    {
                        Debug.Log("Received jump message: " + message);
						rb.velocity += new Vector3(0, 10, 0);
                    });
                }
            }
            catch (Exception e)
            {
                Debug.LogError("UDP error: " + e.Message);
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
