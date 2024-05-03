using DlibDotNet;
using DlibDotNet.Extensions;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Reflection.Metadata;
using System.Text;
using System.Threading.Tasks;

namespace EmoShift
{

    class FaceAnalysis
    {
        private readonly FrontalFaceDetector _detector;
        private readonly ShapePredictor _predictor;
        private const double BlinkRatioThreshold = 3.2;

        public FaceAnalysis(string predictorPath)
        {
            _detector = Dlib.GetFrontalFaceDetector();
            _predictor = ShapePredictor.Deserialize(predictorPath);
        }

        public bool IsBlinking(ref Mat frame, Array2D<BgrPixel> dlibImage, bool showFaceLandmarks=false)
        {
            // These landmarks are based on the image above 
            int[] leftEyeLandmarks = { 36, 37, 38, 39, 40, 41 };
            int[] rightEyeLandmarks = { 42, 43, 44, 45, 46, 47 };

            // Step 2: Convert image to grayscale
            //Mat grayFrame = new Mat();
            //CvInvoke.CvtColor(frame, grayFrame, ColorConversion.Bgr2Gray);

            //Bitmap bitmapGrayFrame = grayFrame.ToBitmap();
            //bitmapGrayFrame.Save("grayframe.png");


            //var dlibImage = Dlib.LoadImage<BgrPixel>("grayframe.png");

            //Image<Bgr, byte> grayFrameImg = frame.ToImage<Bgr, byte>();


            //// Step 3: Face detection with Dlib
            var faces = _detector.Operator(dlibImage);

            // Step 4: Detecting Eyes using landmarks in Dlib
            foreach (var face in faces)
            {
                var landmarks = _predictor.Detect(dlibImage, face);
                if (showFaceLandmarks)
                { 
                    for (uint x = 0; x < landmarks.Parts; x++)
                    {
                        System.Drawing.Point lmPoint = new System.Drawing.Point(landmarks.GetPart(x).X, landmarks.GetPart(x).Y);
                        CvInvoke.DrawMarker(frame, lmPoint, new MCvScalar(255, 100, 100), MarkerTypes.Star, 10, 1);
                        //CvInvoke.PutText(frame, (x+1).ToString(),lmPoint, FontFace.HersheyPlain, 12, new MCvScalar(255,100,100));
                    }
                }

                // Step 5: Calculating blink ratio for one eye
                double leftEyeRatio = GetBlinkRatio(leftEyeLandmarks, landmarks);
                double rightEyeRatio = GetBlinkRatio(rightEyeLandmarks, landmarks);
                double blinkRatio = (leftEyeRatio + rightEyeRatio) / 2;
                if (blinkRatio > BlinkRatioThreshold)
                    return true;
                else
                    return false;
            }
            return false;
        }

        private static double GetBlinkRatio(int[] eyePoints, FullObjectDetection landmarks)
        {
            // Loading all the required points
            DlibDotNet.Point cornerLeft, cornerRight = new DlibDotNet.Point();
            cornerLeft = landmarks.GetPart((uint)eyePoints[0]);
            cornerRight = landmarks.GetPart((uint)eyePoints[3]);

            var centerTop = Midpoint(landmarks.GetPart((uint)eyePoints[1]), landmarks.GetPart((uint)eyePoints[2]));
            var centerBottom = Midpoint(landmarks.GetPart((uint)eyePoints[5]), landmarks.GetPart((uint)eyePoints[4]));

            // Calculating distance
            double horizontalLength = EuclideanDistance(cornerLeft, cornerRight);
            double verticalLength = EuclideanDistance(centerTop, centerBottom);

            double ratio = horizontalLength / verticalLength;

            return ratio;
        }


        // TODO: Move to Helper
        private static DlibDotNet.Point Midpoint(DlibDotNet.Point p1, DlibDotNet.Point p2)
        {
            return new DlibDotNet.Point((p1.X + p2.X) / 2, (p1.Y + p2.Y) / 2);
        }

        private static double EuclideanDistance(DlibDotNet.Point p1, DlibDotNet.Point p2)
        {
            return Math.Sqrt(Math.Pow(p2.X - p1.X, 2) + Math.Pow(p2.Y - p1.Y, 2));
        }


    }



}
