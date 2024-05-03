using DlibDotNet;
using DlibDotNet.Extensions;
using System;
using System.Drawing;
using System.IO;
using Emgu.CV;
using Emgu.Util;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Microsoft.Maui.Controls;
using Emgu.CV.Face;
using Microsoft.Maui.Storage;
using FaceRecognitionDotNet;
using FaceRecognitionDotNet.Extensions;


#if WINDOWS
using Microsoft.UI;
using Microsoft.UI.Windowing;
using Windows.Graphics;
#endif

namespace EmoShift
{
    public partial class MainPage : ContentPage
    {

        bool _recording = false;
        static int fps = 10;

        //TODO: Use MP4, below currently not working?
        //int fourcc = VideoWriter.Fourcc('H', '2', '6', '4');
        //int backend_idx = 0; //any backend;

        VideoCapture _capture = new VideoCapture();
        // TODO: Save this file somewhere with some unique name
        VideoWriter _videoWriter = new VideoWriter("test.avi", fps, new System.Drawing.Size(640, 480), true);

        public MainPage()
        {
            InitializeComponent();
        }

        private void recordVideo() {
            // TODO: grab the dat/xml from kaleidoscope
            CascadeClassifier faceCascade = new CascadeClassifier("hrv/haarcascade_frontalface_default.xml");
            CascadeClassifier handCascade = new CascadeClassifier("hand/haarcascade_hand.xml");

            FaceAnalysis faceAnalysis = new FaceAnalysis("face/shape_predictor_68_face_landmarks.dat");
            YawnAnalysis yawnAnlysis = new YawnAnalysis("hrv/haarcascade_frontalface_default.xml", "face/shape_predictor_68_face_landmarks.dat");

            int frameCounter = 0;

            //using (Mat frame = new Mat())
            //using (_capture = new VideoCapture())
            Mat frame = new Mat();
            _capture = new VideoCapture();
                while (_recording)
                {
                    _capture.Read(frame);

                    string EmoLoglineText = frameCounter.ToString() + ": ";

                    //Reuseable grayframe
                    Mat grayFrame = new Mat();
                    CvInvoke.CvtColor(frame, grayFrame, ColorConversion.Bgr2Gray);
                    Bitmap bitmapGrayFrame = grayFrame.ToBitmap();
                    bitmapGrayFrame.Save("grayframe.png");

                    //Reusable bitmapFrame and arrayImage
                    string tmpImageName = "frame_" + string.Format("{0,6:00000}", frameCounter) + ".png";
                    Bitmap bitmapFrame = frame.ToBitmap();
                    bitmapFrame.Save(tmpImageName);
                    var arrayImage = Dlib.LoadImage<BgrPixel>(tmpImageName);

                    //Do Humate frame by frame analysis Here!

                    // DETECT FACES
                    System.Drawing.Rectangle[] faces = faceCascade.DetectMultiScale(frame, 1.1, 3 , System.Drawing.Size.Empty);
                    // DRAW RECT AROUND FACE
                    System.Drawing.Rectangle[] faceRect = new System.Drawing.Rectangle[faces.Length];
                    int i = 0;
                    foreach (var faceRectTmp in faces) { 
                        CvInvoke.Rectangle(frame, faceRectTmp, new Bgr(System.Drawing.Color.Red).MCvScalar, 2);
                        faceRect[i] = faceRectTmp;
                        i++;
                    }

                    // IS BLINKING
                    EmoLoglineText += "Blinking=" + faceAnalysis.IsBlinking(ref frame, arrayImage, cbShowFaceLandmarks.IsChecked).ToString() + ", ";

                    // IS YAWNING
                    Dictionary<string, string> yawning = yawnAnlysis.IsYawning(frame, grayFrame);
                    EmoLoglineText += ", Eyes=" + yawning["eyes"];
                    EmoLoglineText += ", HeadTilt=" + yawning["head_tilt"];
                    EmoLoglineText += ", Mouth=" + yawning["mouth"];
                    EmoLoglineText += ", MouthAR=" + yawning["mouthAR"];
                    EmoLoglineText += ", Yawn=" + yawning["yawn"];

                    // DETECT HANDS
                    System.Drawing.Size minSlideWindow = new System.Drawing.Size(40,40);    
                    System.Drawing.Rectangle[] hands = handCascade.DetectMultiScale(frame, 1.3, 7, minSlideWindow);
                    // DRAW RECT AROUND HANDS
                    System.Drawing.Rectangle[] handRect = new System.Drawing.Rectangle[hands.Length];
                    i = 0;
                    foreach (var handRectTmp in hands)
                    {
                        CvInvoke.Rectangle(frame, handRectTmp, new Bgr(System.Drawing.Color.Yellow).MCvScalar, 2);
                        handRect[i] = handRectTmp;
                        i++;
                    }
                // HAND MOVEMENT



                //Unique File based


                //FER - Face Emo Recognition
                FaceRecognitionDotNet.Image frImage = FaceRecognition.LoadImage(bitmapFrame);
                using (var fr = FaceRecognition.Create("face/"))
                {
                    using (var estimator = new SimpleEmotionEstimator("emotion/Corrective_re-annotation_of_FER_CK+_KDEF-cnn_300_0.dat"))
                    {
                        var frLocation = fr.FaceLocations(frImage, 1, Model.Cnn).ToArray()[0];
                        fr.CustomEmotionEstimator = estimator;
                        var emotion = fr.PredictEmotion(frImage, frLocation);
                    }
                }

                string tmpImageNameA = "frame_" + string.Format("{0,6:00000}", frameCounter) + ".png";
                    Bitmap bitmapFrameA = frame.ToBitmap();
                    bitmapFrameA.Save(tmpImageNameA);

                    VideoImg.Source = ImageSource.FromFile(tmpImageNameA);


                    //Convert the Mat frame to a Bitmap for display in image control
                    //try
                    //{
                    //    using (Bitmap bitmapFrame = frame.ToBitmap())
                    //    {
                    //        // Create an ImageSource from the Bitmap
                    //        var imageSource = ImageSource.FromStream(() =>
                    //        {
                    //            var imgStream = new MemoryStream();
                    //            //TODO: will need to add OS specific commands for this
                    //            bitmapFrame.Save(stream: imgStream, format: System.Drawing.Imaging.ImageFormat.Png);
                    //            imgStream.Seek(0, SeekOrigin.Begin);
                    //            return imgStream;
                    //        });
                    //        VideoImg.Source = imageSource;
                    //        bitmapFrame.Dispose();
                    //    }
                    //}
                    //catch
                    //{
                    //    continue;
                    //}


                    //Adding frame to video
                    _videoWriter.Write(frame);


                    EmoLoglineText += "\n";
                    EmoLog.Text = EmoLoglineText + EmoLog.Text;

                    // Needs this to interrupt
                    // Checking frame every 1 sec... CACI security doesn't like it going too fast
                    Emgu.CV.CvInvoke.WaitKey(500);




#if WINDOWS
                            Console.WriteLine("WINDOWS");
#endif
#if MACCATALYST
                            Console.WriteLine("MACCATALYST");
#endif
#if ANDROID
                            Console.WriteLine("ANDROID");
#endif
#if IOS
                    Console.WriteLine("IOS");
#endif


                    //if (capture.Fps > 0)
                    //{
                    //    Emgu.CV.CvInvoke.WaitKey((1 / (int)capture.Fps) * 1000);
                    //}
                    //else
                    //{
                    //    Emgu.CV.CvInvoke.WaitKey(1);
                    //}

                    frameCounter++;
                }
        }

        private void OnRecordBtnClicked(object sender, EventArgs e)
        {
            _recording = !_recording;
            if (_recording ) {
                RecordBtn.Text = "Stop Recording";
                recordVideo();
            } else
            {
                RecordBtn.Text = "Start Recording";
                _capture.Dispose();
                _videoWriter.Dispose();
            }
        }

        private void OnEditorTextChanged(object sender, TextChangedEventArgs e)
        {

        }

        private void OnEditorCompleted(object sender, EventArgs e)
        {

        }
    }
}
