package com.example.robert.open_cv_optical_flow;

import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.graphics.Bitmap;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.io.File;
import java.io.FileOutputStream;
import java.util.*;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "CameraActivity";

    private boolean dense = true;

    private Mat mRgba;
    private Mat mIntermediateMat;
    private Mat mGray;
    private Mat mPrevGray;
    private boolean start = false;
    private Mat image;


    MatOfPoint2f prevFeatures, nextFeatures;
    MatOfPoint features;
    List<Point> points;
    Mat prevImage, curImage, flowMat;

    MatOfByte status;
    MatOfFloat err;

    private MenuItem mItemPreviewDense, mItemPreviewSparse;

    private CameraBridgeViewBase mOpenCvCameraView;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        points = new ArrayList<>();
        Button closeButton = findViewById(R.id.start_stop);
        closeButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                start = !start;
                if (!points.isEmpty()) {
                    int p1 = 0;
                    int p2 = 1;
                    Scalar color = new Scalar(255, 255, 255);
                    for (Point p : points) {
                       p.x += 1000;
                       p.y += 1000;
                    }
                    while (p2 != points.size()) {
                        Imgproc.line(image, points.get(p1), points.get(p2), color);
                        p1++;
                        p2++;
                    }
                    Bitmap bmp = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(image, bmp);
                    Long ts = System.currentTimeMillis()/1000;
                    storeBitmap(bmp, ts.toString());
                }

                resetVars();
            }
        });

        mOpenCvCameraView = findViewById(R.id.main_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setMaxFrameSize(240, 144);

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_camera, menu);
        mItemPreviewDense = menu.add("Dense (Farneback)");
        mItemPreviewSparse = menu.add("Sparse (Lucas Kanade)");
        return true;
    }


    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        if (item == mItemPreviewDense) {
            resetVars();
            dense = true;
        } else if (item == mItemPreviewSparse) {
            resetVars();
            dense = false;
        }

        return true;

    }

    private void resetVars() {
        curImage = new Mat();
        prevImage = new Mat();
        flowMat = new Mat();

        mPrevGray = new Mat(mGray.rows(), mGray.cols(), CvType.CV_8UC1);
        image = new Mat(4000, 4000, CvType.CV_8UC1);
        features = new MatOfPoint();
        prevFeatures = new MatOfPoint2f();
        points = new ArrayList<>();
        nextFeatures = new MatOfPoint2f();
        status = new MatOfByte();
        err = new MatOfFloat();
    }

    private void storeBitmap(Bitmap bmp, String image_name) {
        String root = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).toString();
        File myDir = new File(root + "/opencv-optical-flow");
        myDir.mkdirs();
        String fname = "opencv-optical-flow" + image_name+ ".jpg";
        File file = new File(myDir, fname);
        if (file.exists()) file.delete();
        Log.i("LOAD", root + fname);
        try {
            FileOutputStream out = new FileOutputStream(file);
            bmp.compress(Bitmap.CompressFormat.JPEG, 90, out);
            out.flush();
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        // Tell the media scanner about the new file so that it is
        // immediately available to the user.
        MediaScannerConnection.scanFile(this, new String[] { file.toString() }, null,
                new MediaScannerConnection.OnScanCompletedListener() {
                    public void onScanCompleted(String path, Uri uri) {
                        Log.i("ExternalStorage", "Scanned " + path + ":");
                        Log.i("ExternalStorage", "-> uri=" + uri);
                    }
                });
    }


    @Override
    protected void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    // Camera Code

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
        resetVars();
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
        mIntermediateMat.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        return dense ? denseFlow(inputFrame) : sparseFlow(inputFrame);
    }

    public Mat denseFlow(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mGray = inputFrame.gray();
        Mat mGrayT = mGray.t();
        Core.flip(mGray.t(), mGrayT, 1);
        Imgproc.resize(mGrayT, mGrayT, mGray.size());
        if (start) {
            if (prevImage.empty()) {
                prevImage = mGrayT;
            }
            else {
                curImage = mGrayT;
                Video.calcOpticalFlowFarneback(
                    prevImage,
                    curImage,
                    flowMat,
                    0.4,
                    1,
                    8,
                    10,
                    8,
                    1.2,
                    0
                );
                double avgX = 0;
                double avgY = 0;
                int step = 30;
                int numPoints = 0;
                for (int y = 0; y < flowMat.rows(); y += step) {
                    for (int x = 0; x < flowMat.cols(); x += step) {
                        double[] point = flowMat.get(y, x);
                        avgY -= point[1];
                        avgX -= point[0];
                        numPoints++;

                    }
                }
                avgX /= numPoints;
                avgY /= numPoints;
                if (points.isEmpty()) {
                    points.add(new Point(avgX, avgY));
                }
                else {
                    Point lastPoint = points.get(points.size() - 1);
                    points.add(new Point(avgX + lastPoint.x, avgY + lastPoint.y));
                }
                prevImage = curImage;
            }
        }
        return mGrayT;
    }

    public Mat sparseFlow(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mGray = inputFrame.gray();
        Mat mGrayT = mGray.t();
        Core.flip(mGray.t(), mGrayT, 1);
        Imgproc.resize(mGrayT, mGrayT, mGray.size());
        double xAvg1 = 0;
        double xAvg2 = 0;
        double yAvg1 = 0;
        double yAvg2 = 0;

        if (start) {
            if (features.toArray().length == 0) {
                int rowStep = 50, colStep = 100;

                Point points[] = new Point[12];
                int k = 0;
                for (int i = 3; i <= 6; i++) {
                    for (int j = 2; j <= 4; j++) {
                        points[k] = new Point(j * colStep, i * rowStep);
                        k++;
                    }
                }

                features.fromArray(points);

                prevFeatures.fromList(features.toList());
                mPrevGray = mGrayT.clone();
            }

            nextFeatures.fromArray(prevFeatures.toArray());

            Video.calcOpticalFlowPyrLK(
                mPrevGray,
                mGrayT,
                prevFeatures,
                nextFeatures,
                status,
                err
            );

            List<Point> prevList = features.toList(), nextList = nextFeatures.toList();
            Scalar color = new Scalar(255, 0, 0);
            int listSize = prevList.size();

            for (int i = 0; i < listSize; i++) {
                if (prevList.get(i) != null) {
                    xAvg1 += prevList.get(i).x;
                    yAvg1 += prevList.get(i).y;
                }
                if (nextList.get(i) != null) {
                    xAvg2 += nextList.get(i).x;
                    yAvg2 += nextList.get(i).y;
                }
                Imgproc.line(mGrayT, prevList.get(i), nextList.get(i), color);
            }

            xAvg1 /= listSize;
            xAvg2 /= listSize;
            yAvg1 /= listSize;
            yAvg2 /= listSize;
            double pointX = xAvg1 - xAvg2;
            double pointY = yAvg1 - yAvg2;
            if (points.isEmpty()) {
                points.add(new Point(pointX, pointY));
            } else {
                Point lastPoint = points.get(points.size() - 1);
                pointX += lastPoint.x;
                pointY += lastPoint.y;
                points.add(new Point(pointX, pointY));
            }
            mPrevGray = mGrayT.clone();
        }

        return mGrayT;
    }

}
