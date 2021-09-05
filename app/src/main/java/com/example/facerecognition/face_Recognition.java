package com.example.facerecognition;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;

import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class face_Recognition {
    private Interpreter interpreter;

    private  int INPUT_SIZE;
    private int height=0;
    private int width = 0;
    private GpuDelegate gpuDelegate =null; //run model using gpu


    private CascadeClassifier cascadeClassifier;

    face_Recognition(AssetManager assetManager, Context context, String modelPath, int input_size )throws IOException{

        INPUT_SIZE = input_size;
        Interpreter.Options options = new Interpreter.Options();
        gpuDelegate = new GpuDelegate();


        //load model
        //before load add number of threads
        options.setNumThreads(5);
        interpreter = new Interpreter(loadModel(assetManager, modelPath), options);

        //when model is successfully loaded
        Log.d("face_Recognition", "Model is loaded");

        //load haar cascade model
        try{
            //define inpustStream to read haarcascade file
            InputStream inputStream = context.getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
            File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
            //create a new cascade file in that folder
            File mCascadeFile= new File(cascadeDir, "haarcascade_frontalface_alt");
            //define output stream to save haarcascade_frontalface_alt in mCascadeFile

            FileOutputStream outputStream=new FileOutputStream(mCascadeFile);
            //create empty byte buffer to store byte
            byte[] buffer=new byte[4096];
            int byteRead;
            //now read byte in loop
            //when it read -1 that means no data to read
            while((byteRead=inputStream.read(buffer))!=-1){
                outputStream.write(buffer, 0,byteRead);
            }
            //when reading file is complete
            inputStream.close();
            outputStream.close();

            //load cascade classifier
            //                                         path of save file
            cascadeClassifier=new CascadeClassifier(mCascadeFile.getAbsolutePath());
            //if cascade classifier is successfully loaded
            Log.d("face_Recognition", "Classifier is loaded");
            //select device and run

        }
        catch (IOException e){
            e.printStackTrace();
        }

    }

    private MappedByteBuffer loadModel(AssetManager assetManager, String modelPath) throws IOException {


        //this will give description of modelPath
        AssetFileDescriptor assetFileDescriptor = assetManager.openFd(modelPath);
        //create a inputstream to read model path
        FileInputStream inputStream= new FileInputStream(assetFileDescriptor.getFileDescriptor());

        FileChannel fileChannel=inputStream.getChannel();
        long startOffset= assetFileDescriptor.getStartOffset();
        long declaredLength =  assetFileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }

}
