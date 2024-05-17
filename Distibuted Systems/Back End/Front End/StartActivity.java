package com.example.frontend_distributed_systems_2023;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentTransaction;

import android.Manifest;
import android.app.AlertDialog;
import android.app.Dialog;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

public class StartActivity extends AppCompatActivity {

    //Permission request
    private int STORAGE_PERMISSION_CODE = 1;
    private int STORAGE_PERMISSION_CODE_READ = 1;
    private Handler handler = new Handler();
    public String path = Environment.getExternalStorageDirectory().getAbsolutePath();
    //private static final String TAG = "PERMISSION_TAG";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_start);
        FragmentManager fragmentManager = getSupportFragmentManager();
        Button btnprofile = findViewById(R.id.prof_button);
        if (ContextCompat.checkSelfPermission(StartActivity.this,
                Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(StartActivity.this, "You have already granted permission!", Toast.LENGTH_SHORT).show();

        } else {
            requestStoragePermission();
        }
        btnprofile.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                fragmentManager.beginTransaction().replace(R.id.flFragment, ProfileFragment.class, null)
                        .setReorderingAllowed(true)
                        .addToBackStack("name")
                        .commit();
            }
        });
        Button btnstatistics = findViewById(R.id.stats_button);
        btnstatistics.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                fragmentManager.beginTransaction().replace(R.id.flFragment, StatisticsFragment.class, null)
                        .setReorderingAllowed(true)
                        .addToBackStack("name")
                        .commit();
            }
        });
        Button btnsroutes = findViewById(R.id.route_button);
        btnsroutes.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (ContextCompat.checkSelfPermission(StartActivity.this,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(StartActivity.this, "You have already granted permission!", Toast.LENGTH_SHORT).show();

                    copyFileFromAssetsToStorage("route1.gpx");
                    copyFileFromAssetsToStorage("route4.gpx");

                } else {
                    requestStoragePermission();
                }

                fragmentManager.beginTransaction().replace(R.id.flFragment, RoutesFragment.class, null)
                        .setReorderingAllowed(true)
                        .addToBackStack("name")
                        .commit();
            }
        });


    }

    private void requestStoragePermission() {
        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                if (ActivityCompat.shouldShowRequestPermissionRationale(StartActivity.this, android.Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            new AlertDialog.Builder(StartActivity.this)
                                    .setTitle("Permission needed")
                                    .setMessage("This permission is needed to continue")
                                    .setPositiveButton("Allow", new DialogInterface.OnClickListener() {
                                        @Override
                                        public void onClick(DialogInterface dialog, int which) {
                                            ActivityCompat.requestPermissions(StartActivity.this, new String[]{android.Manifest.permission.WRITE_EXTERNAL_STORAGE}, STORAGE_PERMISSION_CODE);
                                        }
                                    })
                                    .setNegativeButton("Deny", new DialogInterface.OnClickListener() {
                                        @Override
                                        public void onClick(DialogInterface dialog, int which) {
                                            dialog.dismiss();
                                        }
                                    }).create().show();
                        }
                    });
                } else {
                    ActivityCompat.requestPermissions(StartActivity.this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, STORAGE_PERMISSION_CODE);
                }
                if (ActivityCompat.shouldShowRequestPermissionRationale(StartActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE)) {
                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            new AlertDialog.Builder(StartActivity.this)
                                    .setTitle("Permission needed")
                                    .setMessage("This permission is needed to read file")
                                    .setPositiveButton("Allow", new DialogInterface.OnClickListener() {
                                        @Override
                                        public void onClick(DialogInterface dialog, int which) {
                                            ActivityCompat.requestPermissions(StartActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, STORAGE_PERMISSION_CODE);
                                        }
                                    })
                                    .setNegativeButton("Deny", new DialogInterface.OnClickListener() {
                                        @Override
                                        public void onClick(DialogInterface dialog, int which) {
                                            dialog.dismiss();
                                        }
                                    }).create().show();
                        }
                    });
                } else {
                    ActivityCompat.requestPermissions(StartActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, STORAGE_PERMISSION_CODE);
                }
            }
        });

        thread.start();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == STORAGE_PERMISSION_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(StartActivity.this, "Permission granted", Toast.LENGTH_SHORT).show();
                    }
                });
            } else {
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(StartActivity.this, "Permission Denied", Toast.LENGTH_SHORT).show();
                    }
                });
            }
        }
        if (requestCode == STORAGE_PERMISSION_CODE_READ) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(StartActivity.this, "Permission granted", Toast.LENGTH_SHORT).show();
                    }
                });
            } else {
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(StartActivity.this, "Permission Denied", Toast.LENGTH_SHORT).show();
                    }
                });
            }
        }
    }
    private void copyFileFromAssetsToStorage(String filename) {
        AssetManager assetManager = getApplicationContext().getAssets();
        InputStream inputStream = null;
        OutputStream outputStream2 = null;
        OutputStream outputStream = null;
        try {
            inputStream = assetManager.open(filename);

            // Read the XML file as a byte array
            byte[] fileBytes = new byte[inputStream.available()];
            inputStream.read(fileBytes);

            // Encode the byte array using UTF-8
            byte[] encodedBytes = null;
            String fileContent = new String(fileBytes, StandardCharsets.UTF_8);
            encodedBytes = fileContent.getBytes(StandardCharsets.UTF_8);

            File storageDir = new File(Environment.getExternalStorageDirectory(), "gpxs");
            if (!storageDir.exists()) {
                storageDir.mkdirs();
            }
            File outputFile = new File(storageDir, filename);

            outputStream = new FileOutputStream(outputFile);

            // Write the encoded byte array to the output file
            outputStream.write(encodedBytes);


            outputStream.flush();
            Toast.makeText(this, "File copied to storage", Toast.LENGTH_SHORT).show();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (inputStream != null) {
                    inputStream.close();
                }
                if (outputStream != null) {
                    outputStream.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }


//    private void copyFileFromAssetsToStorage(String filename) {
//        AssetManager assetManager = getApplicationContext().getAssets();
//        InputStream inputStream = null;
//        OutputStream outputStream = null;
//        try {
//
//            inputStream = assetManager.open(filename);
//            File storageDir = new File(Environment.getExternalStorageDirectory(), "gpxs");
//            if (!storageDir.exists()) {
//                storageDir.mkdirs();
//            }
//            // OutputStream outputStream;
//            File outputFile = new File(storageDir, "route1.gpx");
//            outputStream = new FileOutputStream(outputFile);
//
//            byte[] buffer = new byte[1024];
//            int read;
//            while ((read = inputStream.read(buffer)) != -1) {
//                outputStream.write(buffer, 0, read);
//            }
//
//            outputStream.flush();
//            Toast.makeText(this, "File copied to storage", Toast.LENGTH_SHORT).show();
//        } catch (IOException e) {
//            e.printStackTrace();
//        } finally {
//            try {
//                if (inputStream != null) {
//                    inputStream.close();
//                }
//                if (outputStream != null) {
//                    outputStream.close();
//                }
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
//        }



    }

