package com.example.facerecognition;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Objects;

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
        options.setNumThreads(4);
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
    //class with input mat and output mat
    public Mat recognizeImage(Mat mat_image){
        //rotate map_image by 90 degrees to properly align
        Core.flip(mat_image.t(),mat_image,1);
        //do all processing here & convert image to grayscale

        Mat grayscaleImage= new Mat();

                            //input     output          type
        Imgproc.cvtColor(mat_image, grayscaleImage,Imgproc.COLOR_RGBA2GRAY);
        //now define height and weight
        height=grayscaleImage.height();
        width=grayscaleImage.width();

        //define minimum height and width of face in frame
        int absoluteFaceSize=(int) (height*0.1);
        MatOfRect faces = new MatOfRect();  //this will store all faces

        //check if cascadeclasifier is loaded or not

        if(cascadeClassifier != null){
            //detect face in frame
                                                //input        output   scale of face                       minimum size of face
            cascadeClassifier.detectMultiScale(grayscaleImage, faces, 1.1,2,2,
                    new Size(absoluteFaceSize,absoluteFaceSize),new Size());
        }

        Rect[] faceArray=faces.toArray();

        for (int i=0;i<faceArray.length;i++){
            //draw rectangle around face
                                //input/output, starting point,  endpoint,     color
            Imgproc.rectangle(mat_image, faceArray[i].tl(), faceArray[i].br(), new Scalar(0,255,0,255), 2);

                            //starting x coordinate         starting y coordinate
            Rect roi=new Rect((int)faceArray[i].tl().x,(int)faceArray[i].tl().y,
                    ((int)faceArray[i].br().x)-((int)faceArray[i].tl().x),
                    ((int)faceArray[i].br().y)-((int)faceArray[i].tl().y));

            //roi is used to crop faces
            Mat cropped_rgb=new Mat(mat_image, roi);
            Bitmap bitmap = null;
            bitmap = Bitmap.createBitmap(cropped_rgb.cols(), cropped_rgb.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(cropped_rgb, bitmap);

            //now scale bitmap to model input size 96

            Bitmap scaledBitmap=Bitmap.createScaledBitmap(bitmap,INPUT_SIZE, INPUT_SIZE, false);
            //now convert scaledBitmap to byteBuffer
            ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);

            float[][] face_value=new float[1][1];
            interpreter.run(byteBuffer,face_value);

            //to see face value
            Log.d("face_Recognition", "Out: " + Array.get(Objects.requireNonNull(Array.get(face_value, 0)),0));

            float read_faces=(float) Array.get(Array.get(face_value, 0 ),0);
            String face_name=get_face_name(read_faces);
            //now we will puttext on frame
                             //in/output        //name          starting point                              //ending point
            Imgproc.putText(mat_image, ""+ face_name, new Point((int)faceArray[i].tl().x+10, (int)faceArray[i].tl().y+20),
                    1, 1.5, new Scalar(255,255,255,150),2);

        }

        //rotate by -90
        Core.flip(mat_image.t(), mat_image, 0);

        return mat_image;
    }


    //to be optimized after final output
    private String get_face_name(float read_faces) {
        String val = "";
        if(read_faces>=0 & read_faces < 0.5){
            val="Courtney Cox";
        }
        else if(read_faces>=0.5 & read_faces < 1.5){
            val="Arnold Schwarzenegger";
        }
        else if(read_faces>=1.5 & read_faces < 2.5){
            val="Bhuvan Bam";
        }
        else if(read_faces>=2.5 & read_faces < 3.5){
            val="Hardik Pandya";
        }
        else if(read_faces>=3.5 & read_faces < 4.5){
            val="David Schwimmer";
        }
        else if(read_faces>=4.5 & read_faces < 5.5){
            val="Matt LeBlanc";
        }
        else if(read_faces>=5.5 & read_faces < 6.5){
            val="Simon Helberg";
        }
        else if(read_faces>=6.5 & read_faces < 7.5){
            val="Scarlett Johansson";
        }
        else if(read_faces>=7.5 & read_faces < 8.5){
            val="Pankaj Tripathi";
        }
        else if(read_faces>=8.5 & read_faces < 9.5){
            val="Matthew Perry";
        }
        else if(read_faces>=9.5 & read_faces < 10.5){
            val="Sylvester Stallone";
        }
        else if(read_faces>=10.5 & read_faces < 11.5){
            val="Lionel Messi";
        }
        else if(read_faces>=11.5 & read_faces < 12.5){
            val="Jim Parsons";
        }
        else if(read_faces>=12.5 & read_faces < 13.5){
            val="Not in the dataset";
        }
        else if(read_faces>=13.5 & read_faces < 14.5){
            val="Lisa Kudrow";
        }
        else if(read_faces>=14.5 & read_faces < 15.5){
            val="Muhammad Ali";
        }
        else if(read_faces>=15.5 & read_faces < 16.5){
            val="Brad Pitt";
        }
        else if(read_faces>=16.5 & read_faces < 17.5){
            val="Cristiano Ronaldo";
        }
        else if(read_faces>=17.5 & read_faces < 18.5){
            val="Virat Kohli";
        }
        else if(read_faces>=18.5 & read_faces < 19.5){
            val="Angelina Jolie";
        }
        else if(read_faces>=19.5 & read_faces < 20.5){
            val="Kunal Nayyar";
        }
        else if(read_faces>=20.5 & read_faces < 21.5){
            val="Manoj Bajpayee";
        }
        else if(read_faces>=21.5 & read_faces < 22.5){
            val="Sachin Tendulka";
        }
        else if(read_faces>=22.5 & read_faces < 23.5){
            val="Virat Kohli";
        }
        else if(read_faces>=23.5 & read_faces < 24.5){
            val="Dhoni";
        }
        else if(read_faces>=24.5 & read_faces < 25.5){
            val="Pewdiepie";
        }
        else if(read_faces>=25.5 & read_faces < 26.5){
            val="Aishwarya Rai";
        }
        else if(read_faces>=26.5 & read_faces < 27.5){
            val="Johnny Galeck";
        }
        else if(read_faces>=27.5 & read_faces < 28.5){
            val="Rohis Sharma";
        }
        else{
            val = "Suresh Raina";
        }

        return val;

    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap scaledBitmap) {
        //define ByteBuffer
        ByteBuffer byteBuffer;
        int input_size = INPUT_SIZE;
        //multiply by 4 if input of model is float
        //multiply by 3 is input is RGB
        //if input is gray 3->1
        byteBuffer = ByteBuffer.allocateDirect(4*1*input_size*input_size*3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[input_size*input_size];
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0,0, scaledBitmap.getWidth(),
                scaledBitmap.getHeight());
        int pixels=0;

        //loop through each pixels
        for (int i = 0;i <input_size; ++i){
            for(int j = 0; j< input_size; ++j){
                //each pixel values
                final int val = intValues[pixels++];
                byteBuffer.putFloat((((val>>16)&0xFF))/255.0f);
                byteBuffer.putFloat((((val>>8)&0xFF))/255.0f);
                byteBuffer.putFloat(((val & 0xFF))/255.0f);
                //this thing is important
                //it is placing RGB to MSB to LSB


                //scaling pixels by from 0-255 to 0-1

            }
        }
        return byteBuffer;
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
