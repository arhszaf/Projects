package com.example.frontend_distributed_systems_2023;

import android.annotation.SuppressLint;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.provider.ContactsContract;
import android.util.Log;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.io.StringWriter;
import java.net.Socket;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Map;

public class MyThread extends Thread implements Serializable {
    File file;
    public MyThread(File file) throws UnknownHostException, FileNotFoundException, IOException {
        this.file = file;
    }
    Socket socket = null;
    FileInputStream fileInputStream = null;
    OutputStream outputStream = null;
    ObjectInputStream intermediateResults;
    String fileData;
    static Map<String,Double> map;
    private static final String TAG = "MAP";

    @Override
    public void run() {
        try {
            //
            // Create a new socket connection to your server
            socket = new Socket("192.168.56.1", 6999);
            // Get the input stream of the file
            fileInputStream = new FileInputStream(file);
            outputStream = socket.getOutputStream();
            DataOutputStream dataToSend = new DataOutputStream(outputStream);
            fileData = readFileAsString(file);
            dataToSend.writeUTF(fileData);
            dataToSend.flush();
            outputStream.flush();
           // socket.shutdownOutput();
            Log.d(TAG, "I am here");
            intermediateResults = new ObjectInputStream(socket.getInputStream());
            Log.d(TAG, "now i am here");
            map = (Map<String, Double>) intermediateResults.readObject();
            Log.d(TAG, "Received map: " + map.toString());
            //Log.d(TAG, "I am here ");
//            if(map!=null){
//                Log.d(TAG, "map received from server");
//            }
//            if(map==null){
//                Log.d(TAG, "Wrong data");
//            }

            //Toast.makeText(this, "File sent to server", Toast.LENGTH_SHORT).show();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        } finally {
            try {
                if (fileInputStream != null) {
                    fileInputStream.close();
                }
                if (outputStream != null) {
                    outputStream.close();
                }
                if (intermediateResults != null) {
                    intermediateResults.close();
                }
                if (socket != null) {
                    socket.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
    private byte[] readAllBytes(File file) throws IOException {
        FileInputStream fis = new FileInputStream(file);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buffer = new byte[1024];
        int bytesRead;

//        bytesRead = fis.read(buffer);
//        bos.write(buffer, 0, bytesRead);


        while ((bytesRead = fis.read(buffer)) != -1) {
            bos.write(buffer, 0, bytesRead);
        }

        fis.close();
        bos.close();

        return bos.toByteArray();
    }
    private String readFileAsString(File file) throws IOException {
        StringBuilder fileContent = new StringBuilder();
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        while ((line = reader.readLine()) != null) {
            fileContent.append(line);
            fileContent.append("\n"); // Add line break between lines
        }
        reader.close();
        return fileContent.toString();
    }
//private char[] readAllChars(File file) throws IOException {
//    FileInputStream fis = new FileInputStream(file);
//    InputStreamReader reader = new InputStreamReader(fis, StandardCharsets.UTF_8);
//    StringWriter writer = new StringWriter();
//    char[] buffer = new char[1024];
//    int charsRead;
//
//    while ((charsRead = reader.read(buffer)) != -1) {
//        writer.write(buffer, 0, charsRead);
//    }
//
//    fis.close();
//    reader.close();
//    writer.close();
//
//    String content = writer.toString();
//    return content.toCharArray();
//}




}

//    Handler myHandler;
//
//    String arg;
//
//    public MyThread(String arg, Handler myHandler){
//        this.arg = arg;
//        this.myHandler = myHandler;
//    }
//
//    @Override
//    public void run() {
//        try {
//            Socket s = new Socket("localhost",6999);
//            ObjectOutputStream oos = new ObjectOutputStream(s.getOutputStream());
//            ObjectInputStream ois = new ObjectInputStream(s.getInputStream());
//
//            oos.writeUTF(arg);
//            oos.flush();
//
//            String result = ois.readUTF();
//
//            Message msg = new Message();
//            Bundle bundle = new Bundle();
//            bundle.putString("result",result);
//            msg.setData(bundle);
//            s.close();
//
//            myHandler.sendMessage(msg);
//
//        }catch (Exception e){ }
//    }

