package com.example.frontend_distributed_systems_2023;
import androidx.appcompat.widget.Toolbar;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;


public class MainActivity extends AppCompatActivity {

    private Button toSbutton = null;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        EditText name = findViewById(R.id.firstName);
        EditText lastname = findViewById(R.id.lastName);
        EditText birth = findViewById(R.id.birthDate);
        EditText gender = findViewById(R.id.gender);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
//        set title to toolbar
        toSbutton = (Button) findViewById(R.id.startButton);
       SharedPreferences preferences = getSharedPreferences("MyPrefs", MODE_PRIVATE);
//        toSbutton.setOnClickListener(new View.OnClickListener() {
//          @Override
//          public void onClick(View view) {
//                String getName = name.getText().toString();
//                String getLastName = lastname.getText().toString();
//                String getBirthday = birth.getText().toString();
//                String getGender = gender.getText().toString();
//
//            }
//        });



        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


    }


}