package com.example.frontend_distributed_systems_2023;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;

import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.provider.DocumentsContract;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;


import java.io.*;
import java.net.InetAddress;
import java.net.Socket;
import java.util.Map;


public class RoutesFragment extends Fragment {
    Handler handler;
    TextView label;

    private static final int REQUEST_CODE_FILE_PICKER = 1;

    // Inside your activity
//    private void openFilePicker() {
//        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
//        intent.setType("application/gpx+xml");
//        startActivityForResult(intent, REQUEST_CODE_FILE_PICKER);
//    }





    public RoutesFragment() {
        // Required empty public constructor
    }
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
//        handler = new Handler(Looper.getMainLooper(), new Handler.Callback() {
//            @Override
//            public boolean handleMessage(@NonNull Message message) {
//                String result = message.getData().getString("result");
//
//                label.setText(result);
//
//                return true;
//            }
//        });



    }
//    private Thread connectThread;
//    static Socket requestSocket;
//
//    @Override
//    public void onResume() {
//        super.onResume();
//
//        // Starts the thread
//        connectThread = new Thread(new Runnable() {
//            @Override
//            public void run() {
//                try {
//                    requestSocket = new Socket("127.0.0.1",6999);
//                } catch (IOException e) {
//                    throw new RuntimeException(e);
//                }
//            }
//        });
//        connectThread.start();
//    }
//public void openDirectory(Uri uriToLoad) {
//    // Choose a directory using the system's file picker.
//    Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
//
//    // Optionally, specify a URI for the directory that should be opened in
//    // the system file picker when it loads.
//    intent.putExtra(DocumentsContract.EXTRA_INITIAL_URI, uriToLoad);
//
//    startActivityForResult(intent, your-request-code);
//}


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        View view = inflater.inflate(R.layout.fragment_routes, container, false);
        Button btn_rt1 = view.findViewById(R.id.route_1);
        Button btn_rt2 = view.findViewById(R.id.route_2);
        btn_rt1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                File gpxFile = new File(Environment.getExternalStorageDirectory(), "gpxs/route1.gpx");
                MyThread th1 = null;
                try {
                    th1 = new MyThread(gpxFile);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                th1.start();

            }
        });
        btn_rt2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                File gpxFile = new File(Environment.getExternalStorageDirectory(), "gpxs/route4.gpx");
                MyThread th2 = null;
                try {
                    th2 = new MyThread(gpxFile);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                th2.start();

            }


        });
        return view;
    }
//    private void sendFileToServer(File file) {
//        Socket socket = null;
//        FileInputStream fileInputStream = null;
//        OutputStream outputStream = null;
//        try {
//            // Create a new socket connection to your server
//            socket = new Socket("localhost", 6999);
//
//            // Get the input stream of the file
//           // File file = new File(filePath);
//            fileInputStream = new FileInputStream(file);
//
//            // Get the output stream of the socket
//            outputStream = socket.getOutputStream();
//
//            // Create a buffer to read the file content in chunks
//            byte[] buffer = new byte[1024];
//            int bytesRead;
//
//            // Read the file content and write it to the output stream
//            while ((bytesRead = fileInputStream.read(buffer)) != -1) {
//                outputStream.write(buffer, 0, bytesRead);
//            }
//
//            outputStream.flush();
//            //Toast.makeText(this, "File sent to server", Toast.LENGTH_SHORT).show();
//        } catch (IOException e) {
//            e.printStackTrace();
//        } finally {
//            try {
//                if (fileInputStream != null) {
//                    fileInputStream.close();
//                }
//                if (outputStream != null) {
//                    outputStream.close();
//                }
//                if (socket != null) {
//                    socket.close();
//                }
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
//        }
//    }

}