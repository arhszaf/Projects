# Frontend Distributed Systems
This project is an Android application that connects the frontend with the backend using fragments and socket
communication. It includes multiple activities and fragments to handle different parts of the application. The App is an `activity tracker`. The intefrace consists of 3 buttoms: `Profile`,`Statistics` and `Routes`.The `Profile` of the user displays the name, lastname and date of birth of the user.The `Statistics` compares with the other users based on the total time it took to complete the selected route.The `Routes` displays the available routes to the user to select from.

### Permisions
This application requires the following permissions:
* `READ_EXTERNAL_STORAGE`
* `WRITE_EXTERNAL_STORAGE`
  
These permissions are requested at runtime to ensure the application can read and write files to external storage.
### Usage

#### MainActivity
The `MainActivity` is the entry point of the application. It sets up the toolbar and initializes shared preferences.

#### StartActivity
The `StartActivity` handles the main functionalities of the app, including fragment transitions and permission requests. It includes buttons to navigate to different fragments:

* **ProfileFragment**
* **StatisticsFragment**
* **RoutesFragment**

It also includes logic to copy files from assets to external storage.

#### ProfileFragment
The `ProfileFragment` displays user profile information. It inflates the `fragment_profile` layout.

#### MyThread
The `MyThread` class handles the socket communication with the backend server. It reads a file, sends its content to the server, and receives a map as a response.

#### EndActivity
The `EndActivity` is a simple activity that allows the user to return to the `StartActivity`.
